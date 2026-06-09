import cv2
import time
'''
cap = cv2.VideoCapture(0)  # Usa solo 0 aquí

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
else:
    print("✅ Cámara abierta correctamente")

ret, frame = cap.read()
time.sleep(1)
print("ret: ", ret)
print("frame: ", frame)

cap.release()
'''
import cv2
import time

# Forzamos V4L2 (más fiable en Pi) e indicamos un formato soportado
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  16384)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 16384)
# FourCC para YUYV:
fourcc = cv2.VideoWriter_fourcc('Y','U','Y','V')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

print("cap.isOpened() =", cap.isOpened())
print("Resolución:", cap.get(cv2.CAP_PROP_FRAME_WIDTH),
      "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FOURCC:", cap.get(cv2.CAP_PROP_FOURCC))

# Intentamos leer varios frames
for i in range(5):
    ret, frame = cap.read()
    print(f"Iter {i} — ret:", ret, " frame shape:", 
          None if frame is None else frame.shape)
    time.sleep(0.1)

cap.release()
