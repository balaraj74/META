import os
import sys
import subprocess

try:
    from huggingface_hub import HfApi, login
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print("Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import HfApi, login
    from huggingface_hub.utils import RepositoryNotFoundError

def deploy_to_huggingface():
    print("🚀 TRIAGE Multi-Agent Hospital System — HuggingFace Spaces Deployment\n")
    
    # 1. Check for token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Please enter your Hugging Face write token.")
        print("You can get one at: https://huggingface.co/settings/tokens")
        token = input("HF Token: ").strip()
        login(token)
    
    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    
    repo_name = "triage-multi-agent-system"
    repo_id = f"{username}/{repo_name}"
    
    print(f"\nTargeting Space: {repo_id}")
    
    # 2. Ensure repository exists
    try:
        api.repo_info(repo_id, repo_type="space")
        print("✅ Space already exists.")
    except RepositoryNotFoundError:
        print("Creating new Space...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False
        )
        print("✅ Created new public Space!")
    except Exception as e:
        print(f"Error checking repo: {e}")
        return

    # 3. Upload files
    spaces_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "spaces")
    
    print(f"\nUploading files from {spaces_dir}...")
    api.upload_folder(
        folder_path=spaces_dir,
        repo_id=repo_id,
        repo_type="space",
    )
    
    print("\n🎉 Deployment Complete!")
    print(f"View your live app here: https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    deploy_to_huggingface()
