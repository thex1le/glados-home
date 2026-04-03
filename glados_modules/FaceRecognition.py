"""Face recognition and emotion detection using InsightFace and HSEmotion.

Provides identity tracking (who is this person?) and emotional awareness
(are they happy, angry, etc.) for the GLaDOS vision pipeline.

Face detection + ArcFace embedding runs via InsightFace's FaceAnalysis model.
Emotion classification runs via HSEmotion on face crops.
Known face embeddings are stored in a JSON database on disk.

Only used on the head camera — side cameras have 160-degree fisheye distortion
that makes face recognition unreliable.
"""

# builtin
import json
import traceback
from os import path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# 3rd party
import numpy as np

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (
    VisionResultsEnum, FaceEnums, LoggingEnums
)


class GladosFaceRecognition:
    """Face detection, embedding, identity lookup, and emotion detection.

    Uses InsightFace's buffalo_l model for face detection + ArcFace embedding.
    Uses HSEmotion for facial emotion classification.
    Stores known face embeddings in a JSON database file.

    Args:
        db_path: Path to the face database JSON file.
        similarity_threshold: Cosine similarity threshold for face matching.
    """

    def __init__(self, db_path: str = None,
                 similarity_threshold: float = None,
                 enable_emotion: bool = True) -> None:
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(
            name=self.__name__,
            console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self._db_path = db_path or FaceEnums.FACE_DB_PATH.value
        self._threshold = similarity_threshold or FaceEnums.SIMILARITY_THRESHOLD.value
        self._max_embeddings = FaceEnums.MAX_EMBEDDINGS_PER_PERSON.value
        self._database: Dict[str, dict] = {}
        self._enrollment_pending: Optional[str] = None

        # Initialize InsightFace
        try:
            from insightface.app import FaceAnalysis
            self._face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info("InsightFace model loaded (buffalo_l)")
        except Exception as e:
            self.logger.error(f"Failed to load InsightFace: {traceback.format_exc()}")
            self._face_app = None

        # Initialize HSEmotion (toggleable)
        if enable_emotion:
            try:
                from hsemotion.facial_emotions import HSEmotionRecognizer
                self._emotion_model = HSEmotionRecognizer(
                    model_name="enet_b0_8_best_afew", device="cuda")
                self.logger.info("HSEmotion model loaded")
            except Exception as e:
                self.logger.error(f"Failed to load HSEmotion: {traceback.format_exc()}")
                self._emotion_model = None
        else:
            self.logger.info("Emotion detection disabled via config")
            self._emotion_model = None

        self.load_database()

    def detect_and_analyze(self, image: np.ndarray) -> List[dict]:
        """Detect faces, compute embeddings, identify, and classify emotion.

        Args:
            image: BGR image (from OpenCV/YOLO pipeline).

        Returns:
            List of face dicts, each with:
                - bbox: dict with x1, y1, x2, y2
                - embedding: 512-dim numpy array
                - face_id: str (name or "unknown")
                - face_confidence: float (cosine similarity)
                - emotion: str (e.g., "happy", "neutral")
                - emotion_scores: dict of emotion -> confidence
        """
        if not self._face_app:
            return []

        try:
            faces = self._face_app.get(image)
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.normed_embedding

            # Identify against database
            face_id, similarity = self.identify(embedding)

            # Handle pending enrollment
            if self._enrollment_pending and face_id == "unknown":
                self.enroll(self._enrollment_pending, embedding)
                face_id = self._enrollment_pending
                similarity = 1.0
                self.logger.info(f"Enrolled face: {face_id}")
                self._enrollment_pending = None

            # Emotion detection
            emotion = "neutral"
            emotion_scores = {}
            if self._emotion_model:
                try:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    # Ensure valid crop dimensions
                    h, w = image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        face_crop = image[y1:y2, x1:x2]
                        emotion, scores = self._emotion_model.predict_emotions(
                            face_crop, logits=False)
                        emotion_labels = ["angry", "disgust", "fear", "happy",
                                          "sad", "surprise", "neutral"]
                        if isinstance(scores, np.ndarray):
                            emotion_scores = {
                                label: round(float(s), 3)
                                for label, s in zip(emotion_labels, scores)
                            }
                            emotion = emotion_labels[int(np.argmax(scores))]
                except Exception as e:
                    self.logger.debug(f"Emotion detection failed: {e}")

            face_key = VisionResultsEnum.VISION_RESULTS_FACE_KEY.value
            results.append({
                VisionResultsEnum.VISION_RESULTS_FACE_BOX_KEY.value: {
                    VisionResultsEnum.BOX_X1.value: int(bbox[0]),
                    VisionResultsEnum.BOX_Y1.value: int(bbox[1]),
                    VisionResultsEnum.BOX_X2.value: int(bbox[2]),
                    VisionResultsEnum.BOX_Y2.value: int(bbox[3]),
                },
                "embedding": embedding,
                VisionResultsEnum.VISION_RESULTS_FACE_ID_KEY.value: face_id,
                VisionResultsEnum.VISION_RESULTS_FACE_CONFIDENCE_KEY.value: round(float(similarity), 3),
                VisionResultsEnum.VISION_RESULTS_FACE_DETECTED_KEY.value: True,
                VisionResultsEnum.VISION_RESULTS_EMOTION_KEY.value: emotion,
                VisionResultsEnum.VISION_RESULTS_EMOTION_SCORES_KEY.value: emotion_scores,
            })

        return results

    def match_faces_to_persons(self, faces: List[dict],
                                person_objects: List[dict]) -> None:
        """Assign face detections to YOLO person bounding boxes.

        Matches by checking if the face bbox center falls inside the person bbox.
        Modifies person_objects in place, adding a "face" key to matched persons.

        Args:
            faces: List of face dicts from detect_and_analyze().
            person_objects: List of person detection dicts (with "box" key).
        """
        face_key = VisionResultsEnum.VISION_RESULTS_FACE_KEY.value
        box_key = VisionResultsEnum.VISION_RESULTS_BOX_KEY.value

        for face in faces:
            face_box = face[VisionResultsEnum.VISION_RESULTS_FACE_BOX_KEY.value]
            face_cx = (face_box[VisionResultsEnum.BOX_X1.value] +
                       face_box[VisionResultsEnum.BOX_X2.value]) / 2
            face_cy = (face_box[VisionResultsEnum.BOX_Y1.value] +
                       face_box[VisionResultsEnum.BOX_Y2.value]) / 2

            for person in person_objects:
                person_box = person.get(box_key, {})
                if not person_box:
                    continue
                # Check if face center is inside person bbox
                if (person_box.get(VisionResultsEnum.BOX_X1.value, 0) <= face_cx <=
                        person_box.get(VisionResultsEnum.BOX_X2.value, 0) and
                        person_box.get(VisionResultsEnum.BOX_Y1.value, 0) <= face_cy <=
                        person_box.get(VisionResultsEnum.BOX_Y2.value, 0)):
                    # Don't include the raw embedding in the MQTT message
                    face_data = {k: v for k, v in face.items() if k != "embedding"}
                    person[face_key] = face_data
                    break  # one face per person

    def identify(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Look up a face embedding against the database.

        Args:
            embedding: 512-dim normalized ArcFace embedding.

        Returns:
            Tuple of (face_id, similarity). Returns ("unknown", 0.0) if no match.
        """
        best_id = "unknown"
        best_sim = 0.0

        for name, entry in self._database.items():
            avg_emb = np.array(entry["average_embedding"])
            similarity = float(np.dot(embedding, avg_emb))
            if similarity > best_sim:
                best_sim = similarity
                best_id = name

        if best_sim < self._threshold:
            return "unknown", 0.0
        return best_id, best_sim

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        """Add or update a face in the database.

        Stores up to MAX_EMBEDDINGS_PER_PERSON embeddings per person and
        maintains a running average for matching.

        Args:
            name: Name to associate with this face.
            embedding: 512-dim normalized ArcFace embedding.
        """
        name = name.lower().strip()
        if name in self._database:
            entry = self._database[name]
            embeddings = entry["embeddings"]
            if len(embeddings) >= self._max_embeddings:
                embeddings.pop(0)
            embeddings.append(embedding.tolist())
            entry["average_embedding"] = np.mean(
                np.array(embeddings), axis=0).tolist()
            entry["enrollment_count"] = len(embeddings)
        else:
            self._database[name] = {
                "embeddings": [embedding.tolist()],
                "average_embedding": embedding.tolist(),
                "enrollment_count": 1,
            }

        self._database[name]["last_seen"] = datetime.now().isoformat()
        self.save_database()
        self.logger.info(f"Enrolled face '{name}' "
                         f"({self._database[name]['enrollment_count']} embeddings)")

    def forget(self, name: str) -> bool:
        """Remove a face from the database.

        Args:
            name: Name to remove.

        Returns:
            True if the face was found and removed.
        """
        name = name.lower().strip()
        if name in self._database:
            del self._database[name]
            self.save_database()
            self.logger.info(f"Forgot face '{name}'")
            return True
        return False

    def start_enrollment(self, name: str) -> None:
        """Set enrollment pending — next detected face will be enrolled.

        Args:
            name: Name to enroll the next face as.
        """
        self._enrollment_pending = name.lower().strip()
        self.logger.info(f"Enrollment pending for '{self._enrollment_pending}'")

    def save_database(self) -> None:
        """Persist the face database to disk."""
        try:
            with open(self._db_path, 'w') as f:
                json.dump(self._database, f, indent=2)
            self.logger.debug(f"Face database saved to {self._db_path}")
        except Exception as e:
            self.logger.error(f"Failed to save face database: {e}")

    def load_database(self) -> None:
        """Load the face database from disk."""
        if not path.exists(self._db_path):
            self.logger.info("No face database found, starting fresh")
            return
        try:
            with open(self._db_path, 'r') as f:
                self._database = json.load(f)
            self.logger.info(f"Face database loaded: {len(self._database)} faces")
        except Exception as e:
            self.logger.error(f"Failed to load face database: {e}")

    def get_known_names(self) -> List[str]:
        """Return list of all enrolled face names."""
        return list(self._database.keys())
