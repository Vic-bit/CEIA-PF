import numpy as np
import cv2 as cv 
import glob 
import os
import matplotlib.pyplot as plt
form sift_rpi.config import CALIB_PATH

def calibrate(showPics=True):
    # Read Image
    root = os.getcwd()
    calibrationDir = os.path.join(root, CALIB_PATH)
    imgPathList = glob.glob(os.path.join(calibrationDir,'*.jpeg'))

    # Initialize
    nRows = 7
    nCols = 7
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)
    worldPtsCur = np.zeros ((nRows*nCols, 3), np.float32)
    worldPtsCur [:,: 2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1,2)
    worldPtsList = []
    imgPtsList = []

    # Find Corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners (imgGray, (nRows,nCols), None)

        if cornersFound == True:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray,cornersOrg, (11,11), (-1,-1), termCriteria)
            imgPtsList.append(cornersRefined)
            if showPics:
                cv.drawChessboardCorners(imgBGR, (nRows,nCols), cornersRefined, cornersFound)
                cv.imshow( 'Chessboard', imgBGR)
                cv.waitKey(500)

    cv.destroyAllWindows()

    if len(imgPtsList) == 0:
        raise ValueError("No se detectaron esquinas de tablero en ninguna imagen.")

    # Calibrate
    repError, camMatrix, distCoeff,rvecs, tvecs = cv.calibrateCamera(worldPtsList, 
                                                                     imgPtsList, 
                                                                     imgGray.shape[::-1], 
                                                                     None, 
                                                                     None)
    print( 'Camera Matrix: \n', camMatrix)
    print ("Reproj Error (pixels): {:.4f}". format(repError))

    # Save Calibration Parameters (later video)
    curFolder = os. path. dirname(os.path.abspath(__file__))
    paramPath = os. path.join(curFolder, 'calibration.npz')
    np.savez(paramPath,
    repError=repError,
    camMatrix=camMatrix,
    distCoeff=distCoeff,
    rvecs=rvecs,
    tvecs=tvecs)

    return camMatrix, distCoeff


def runCalibration():
    calibrate(showPics=True)

#def runRemoveDistortion():
#    camMatrix,distCoeff = calibrate(showPics=False)
#    removeDistortion (camMatrix, distCoeff)

if __name__ == '__main__':
    runCalibration()
    # runRemdveDistortion()