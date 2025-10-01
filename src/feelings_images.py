#!/usr/bin/env python3

from deepface import DeepFace
from PIL import Image
import torch
import clip
import cv2
from collections import Counter
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# --- Configuración ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

emotion_labels = ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust", "calm", "romantic", "dark", "party"]

# --- Función CLIP para emoción de escena ---
def predict_scene_emotion(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([f"this image looks {emo}" for emo in emotion_labels]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logits_per_image = (100.0 * image_features @ text_features.T)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    best_idx = probs.argmax()
    return emotion_labels[best_idx], float(probs[best_idx])

# --- Análisis de imagen ---
def analyze_image(image_path):
    results = {"faces": None, "scene": None, "group_emotion": None}
    img_cv = cv2.imread(image_path)

    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        face_emotions = []

        if isinstance(analysis, list):
            for face in analysis:
                x, y, w, h = face["region"].values()
                emotion = face["dominant_emotion"]
                face_emotions.append(emotion)
                cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_cv, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            results["faces"] = face_emotions
            counter = Counter(face_emotions)
            group_emotion, count = counter.most_common(1)[0]
            results["group_emotion"] = {"emotion": group_emotion, "count": count, "total_faces": len(face_emotions)}

        else:
            x, y, w, h = analysis["region"].values()
            emotion = analysis["dominant_emotion"]
            results["faces"] = [emotion]
            results["group_emotion"] = {"emotion": emotion, "count": 1, "total_faces": 1}
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_cv, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    except Exception:
        results["faces"] = None
        results["group_emotion"] = None

    scene_emotion, scene_score = predict_scene_emotion(image_path)
    results["scene"] = {"emotion": scene_emotion, "score": scene_score}
    cv2.putText(img_cv, f"Scene: {scene_emotion} ({scene_score:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    output_image_path = "result_image.jpg"
    cv2.imwrite(output_image_path, img_cv)
    results["output_image"] = output_image_path

    return results

# --- Configuración Spotify ---
client_id = "badb20eb017c467793a4d40622e4c72c"        
client_secret = "2750e9bd69dc4412a2ed282452471b40" 

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

def get_spotify_tracks(emotion, limit=5):
    query = f"{emotion} mood"
    try:
        results = sp.search(q=query, type="playlist", limit=1)
        if not results["playlists"]["items"]:
            return []

        playlist_id = results["playlists"]["items"][0]["id"]
        tracks = sp.playlist_tracks(playlist_id, limit=limit)
        songs = []
        for item in tracks["items"]:
            track = item["track"]
            if track:
                songs.append({
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "url": track["external_urls"]["spotify"]
                })
        return songs
    except Exception:
        return []

# --- Función principal ---
if __name__ == "__main__":
    img_path = "/home/angel/Downloads/party.jpeg" 
    output = analyze_image(img_path)

    scene_emotion = output["scene"]["emotion"]
    scene_score = output["scene"]["score"]
    group_emotion = output["group_emotion"]["emotion"] if output["group_emotion"] else None
    group_faces = output["group_emotion"]["total_faces"] if output["group_emotion"] else 0

    # Ponderar emoción de grupo si hay 2 o más caras
    if group_emotion and group_faces >= 2:
        final_emotion = group_emotion
    else:
        final_emotion = scene_emotion

    print(f"\nDetected feeling: {final_emotion}")
    print(f"Image saved as: {output['output_image']}\n")
    print("Spotify recommendations:\n")

    songs = get_spotify_tracks(final_emotion)
    if songs:
        for s in songs:
            print(f" - {s['name']} – {s['artist']} ({s['url']})")
    else:
        print("No Spotify songs found for this emotion.")
