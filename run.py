from app import app
from get_model import download_model

if __name__ == "__main__":
    download_model()
    app.run()
