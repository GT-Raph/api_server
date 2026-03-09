from fastapi import FastAPI, File, UploadFile, Form, Depends, Header, HTTPException, status, Request
from fastapi.responses import FileResponse
import cv2
import numpy as np
import os
import ulid
import logging
from datetime import datetime
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

try:
    from .config import CAPTURED_FACES_DIR, API_KEY
    from .db_utils import (
        get_db,
        save_snapshot_to_db,
        ensure_tables_exist,
        get_embeddings_db,
        create_processing_job,
        mark_job_completed,
        mark_job_failed
    )
    from .face_utils import match_face_id, enhance_face
except:
    from config import CAPTURED_FACES_DIR, API_KEY
    from db_utils import (
        get_db,
        save_snapshot_to_db,
        ensure_tables_exist,
        get_embeddings_db,
        create_processing_job,
        mark_job_completed,
        mark_job_failed
    )
    from face_utils import match_face_id, enhance_face


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("face_api")

app = FastAPI(title="Bank Face + Emotion API")

os.makedirs(CAPTURED_FACES_DIR, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=4)

# =====================================================
# MODEL PRELOAD
# =====================================================

logger.info("Loading DeepFace models...")

emotion_model = DeepFace.build_model("Emotion")
embedding_model = DeepFace.build_model("Facenet")

logger.info("Models loaded.")


# =====================================================
# API SECURITY
# =====================================================

async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
):

    if not API_KEY:
        return True

    if x_api_key == API_KEY:
        return True

    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        if token == API_KEY:
            return True

    raise HTTPException(status_code=401, detail="Unauthorized")


# =====================================================
# IMAGE SERVING
# =====================================================

@app.get("/images/{filename}", name="serve_image")
async def serve_image(filename: str):

    safe = os.path.basename(filename)
    path = os.path.join(CAPTURED_FACES_DIR, safe)

    if os.path.exists(path):
        return FileResponse(path)

    raise HTTPException(404, "Image not found")


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


# =====================================================
# EMOTION SMOOTHING FUNCTION
# =====================================================

def smooth_emotion_prediction(face_img):

    emotion_vectors = []

    # create small frame variations
    frames = []

    frames.append(face_img)

    h, w = face_img.shape[:2]

    frames.append(face_img[0:h-5, 0:w-5])
    frames.append(face_img[5:h, 5:w])
    frames.append(face_img[2:h-2, 2:w-2])
    frames.append(face_img)

    for frame in frames:

        result = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            models={"emotion": emotion_model},
            enforce_detection=False
        )

        emotion_vectors.append(result[0]["emotion"])

    # average emotions
    avg = {}

    for emotion in emotion_vectors[0].keys():

        values = [vec[emotion] for vec in emotion_vectors]
        avg[emotion] = sum(values) / len(values)

    dominant = max(avg, key=avg.get)

    confidence = avg[dominant]

    return dominant, confidence, avg


# =====================================================
# BACKGROUND PROCESSING
# =====================================================

def process_job(job):

    db = get_db()
    cursor = db.cursor()

    try:

        logger.info(f"Processing job {job['job_id']}")

        frame = cv2.imread(job["image_path"])

        faces = DeepFace.extract_faces(
            img_path=frame,
            enforce_detection=False
        )

        if not faces:
            raise Exception("No face detected")

        known_embeddings = get_embeddings_db(cursor)

        for face in faces:

            fa = face["facial_area"]
            x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

            face_img = frame[y:y+h, x:x+w]

            face_img = enhance_face(face_img)

            # embedding
            emb = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet",
                model=embedding_model,
                enforce_detection=False
            )

            embedding = emb[0]["embedding"]

            match = match_face_id(embedding, known_embeddings)

            face_id = match if match else str(ulid.new())

            # emotion smoothing
            emotion, confidence, emotion_vector = smooth_emotion_prediction(face_img)

            timestamp = datetime.now()

            save_snapshot_to_db(
                db=db,
                face_id=face_id,
                pc_name=job["pc_name"],
                image_path=job["image_path"],
                timestamp=timestamp,
                embedding=embedding,
                emotion=emotion,
                confidence=confidence,
                emotion_vector=emotion_vector
            )

        mark_job_completed(db, job["job_id"])

    except Exception as e:

        logger.error(f"Job failed {job['job_id']} : {str(e)}")

        mark_job_failed(db, job["job_id"], str(e))

    finally:

        cursor.close()
        db.close()


# =====================================================
# UPLOAD ENDPOINT
# =====================================================

@app.post("/upload-face")
async def upload_face(
    request: Request,
    file: UploadFile = File(...),
    pc_name: str = Form(...),
    _auth=Depends(verify_api_key)
):

    try:

        data = await file.read()

        arr = np.frombuffer(data, np.uint8)

        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(400, "Invalid image")

        job_id = str(ulid.new())

        filename = f"{job_id}.jpg"

        path = os.path.join(CAPTURED_FACES_DIR, filename)

        cv2.imwrite(path, frame)

        db = get_db()

        ensure_tables_exist(db)

        create_processing_job(
            db=db,
            job_id=job_id,
            pc_name=pc_name,
            image_path=path,
            timestamp=datetime.now()
        )

        db.close()

        executor.submit(process_job, {
            "job_id": job_id,
            "pc_name": pc_name,
            "image_path": path
        })

        logger.info(f"Job accepted {job_id}")

        return {
            "status": "accepted",
            "job_id": job_id
        }

    except Exception as e:

        logger.error(str(e))

        raise HTTPException(500, "Processing error")
