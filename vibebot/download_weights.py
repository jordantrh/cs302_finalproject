import gdown
import os
import requests

# require pip-system-certs package to be installed

def download_weights():
    # make weights directory if it doesn't exist
    if not os.path.exists('weights'):
        os.makedirs('weights')
    
    # download yolov8s.pt if it doesn't exist
    if not os.path.exists('weights/yolov8s-seg.pt'):
        url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt'
        r = requests.get(url, allow_redirects=True)    
        open('weights/yolov8s-seg.pt', 'wb').write(r.content)

    # download osnet_x0_25_msmt17.pt if it doesn't exist
    if not os.path.exists('weights/osnet_x0_25_msmt17.pt'):
        gdown.download('https://drive.google.com/uc?id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e', 'weights/osnet_x0_25_msmt17.pt', quiet=False)

if __name__ == '__main__':
    download_weights()