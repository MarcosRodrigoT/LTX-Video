import subprocess
from huggingface_hub import snapshot_download


def download_model(model_path):
    snapshot_download("Lightricks/LTX-Video", local_dir=model_path, local_dir_use_symlinks=False, repo_type="model")


if __name__ == "__main__":
    MODEL_PATH = "/mnt/Data/mrt/LTX-Video"

    # Download the model
    download_model(model_path=MODEL_PATH)

    # Set inference parameters
    model_path = MODEL_PATH
    # prompt = (
    #     "A video from a drone point of view flying straight over a rugged mountain range with jagged peaks and valleys, partially covered in dense forests and areas without trees. At the center of"
    #     " the scene, a raging wildfire spreads in a circular pattern, glowing bright orange and red against the dark green forest. Thick plumes of smoke rise into the overcast sky,"
    #     " partially obscuring the sun. The dramatic intensity of the wildfire contrasts with the natural beauty of the mountain landscape, emphasizing the scale and impact of the"
    #     " fire. Images are captured from a top-down view from an altitude of 2 kilometers. The video finishes after the drone has passed straight over the wildfire, capturing images from straight above. The drone flies very close to the fire. The fire and the smoke both flow with the wind."
    # )
    prompt = "A woman with blood on her face and a white tank top looks down and to her right, then back up as she speaks. She has dark hair pulled back, light skin, and her face and chest are covered in blood. The camera angle is a close-up, focused on the woman's face and upper torso. The lighting is dim and blue-toned, creating a somber and intense atmosphere. The scene appears to be from a movie or TV show."
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    height = 480
    width = 704
    num_frames = 129
    frame_rate = 25
    num_inference_steps = 40
    guidance_scale = 3.0
    seed = 99
    output_path = "outputs/"

    # Run inference
    command = [
        "python",
        "inference.py",
        "--ckpt_dir", model_path,
        "--prompt", prompt,
        "--negative_prompt", negative_prompt,
        "--height", str(height),
        "--width", str(width),
        "--num_frames", str(num_frames),
        "--frame_rate", str(frame_rate),
        "--num_inference_steps", str(num_inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--seed", str(seed),
        "--output_path", output_path,
    ]

    subprocess.run(command)
