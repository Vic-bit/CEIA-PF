# camera_calib.py
import numpy as np
import cv2 as cv 
import glob 
import os
import matplotlib.pyplot as plt
from config import CALIB_PATH

def calibrate(showPics=True):
    """
    Realiza la calibración de la cámara utilizando imágenes de un tablero de ajedrez.
    """
    if os.path.isabs(CALIB_PATH):
        calibrationDir = CALIB_PATH
    else:
        calibrationDir = os.path.join(os.getcwd(), CALIB_PATH)
    
    # Check if directory exists
    if not os.path.exists(calibrationDir):
        raise ValueError(f"Directorio de calibración no encontrado: {calibrationDir}")
    
    # Try to find images with different extensions
    imgPathList = []
    for ext in ['*.jpeg', '*.jpg', '*.png', '*.bmp']:
        imgPathList.extend(glob.glob(os.path.join(calibrationDir, ext)))
        imgPathList.extend(glob.glob(os.path.join(calibrationDir, ext.upper())))
    
    print(f"Directorio de calibración: {calibrationDir}")
    print(f"Imágenes encontradas: {len(imgPathList)}")
    if imgPathList:
        print(f"Primeras imágenes: {imgPathList[:3]}")
    
    if len(imgPathList) == 0:
        raise ValueError(f"No se encontraron imágenes en: {calibrationDir}\n"
                        f"Extensiones buscadas: .jpeg, .jpg, .png, .bmp")

    # Initialize
    nRows = 7
    nCols = 7
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows*nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
    worldPtsList = []
    imgPtsList = []

    # Find Corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        if imgBGR is None:
            print(f"Advertencia: No se pudo leer la imagen {curImgPath}")
            continue
            
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows, nCols), None)

        if cornersFound:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            imgPtsList.append(cornersRefined)
            if showPics:
                cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefined, cornersFound)
                cv.imshow('Chessboard', imgBGR)
                cv.waitKey(500)

    cv.destroyAllWindows()

    if len(imgPtsList) == 0:
        raise ValueError("No se detectaron esquinas de tablero en ninguna imagen.")

    # Calibrate
    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(worldPtsList, 
                                                                     imgPtsList, 
                                                                     imgGray.shape[::-1], 
                                                                     None, 
                                                                     None)
    print('Camera Matrix: \n', camMatrix)
    print("Reproj Error (pixels): {:.4f}".format(repError))

    # Save Calibration Parameters
    curFolder = os.path.dirname(os.path.abspath(__file__))
    calibDir = os.path.join(curFolder, 'calibration')
    os.makedirs(calibDir, exist_ok=True)
    paramPath = os.path.join(calibDir, 'calibration.npz')
    np.savez(paramPath,
             repError=repError,
             camMatrix=camMatrix,
             distCoeff=distCoeff,
             rvecs=rvecs,
             tvecs=tvecs)
    print(f"Calibración guardada en: {paramPath}")

    return camMatrix, distCoeff


def runCalibration():
    calibrate(showPics=True)

#def runRemoveDistortion():
#    camMatrix,distCoeff = calibrate(showPics=False)
#    removeDistortion (camMatrix, distCoeff)

if __name__ == '__main__':
    runCalibration()
    # runRemdveDistortion()