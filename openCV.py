import cv2

imagem = cv2.imread('Teste.jpg')

largura_desejada = 800
altura_desejada = int(imagem.shape[0] * (largura_desejada / imagem.shape[1]))
imagem_redimensionada = cv2.resize(imagem, (largura_desejada, altura_desejada))

imagem_cinza = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(imagem_cinza, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(imagem_redimensionada, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Rostos detectados', imagem_redimensionada)
cv2.waitKey(0)
cv2.destroyAllWindows()
