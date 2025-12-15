# display.py
import sdl2
import sdl2.ext
import cv2

class Display(object):
    def __init__(self, W, H):
        sdl2.ext.init()
        self.window = sdl2.ext.Window("SLAM", size=(W,H))
        self.window.show()
        self.W, self.H, W, H

    def paint(self, img):
        # Redimensionar al W y H de la clase
        img = cv2.resize(img, (self.W, self.H))

        # Recibe una lista de SLD2 events
        events = sdl2.ext.get_events()

        for event in events:
            # Se fija si el evento es SDL_QUIT (el evento cierra la ventana)
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        # Recupera el arreglo 3D que representa los datos del pixel y la superficie
        # de la imagen
        surf = sdl2.ext.pixels3D(self.window.get_surface())

        # Actualiza los pixeles de la superficie de la ventana con la imagen redimensionada
        # Intercambia los ejes del array de imagen para que encajen con el formato
        # esperado de SDL
        surf[:,:,0:3] = img.swapaxes(0,1)

        # Refrescar la ventana para actualizar la superficie
        self.window.refresh()
        
