import cv2
import easyocr


def resize_image(image, scale_percent=150):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # Увеличение размера изображения
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('resized', resized)
    # cv2.waitKey(0)  # Ожидание нажатия любой клавиши для продолжения
    # cv2.destroyAllWindows()  # Закрытие всех окон
    return resized


# Функция для предобработки изображения
def preprocess_image(image_path):
    # Чтение изображения
    image = cv2.imread(image_path)
    resized_image = resize_image(image, scale_percent=150)
    return resized_image


# Функция для распознавания текста с использованием EasyOCR
def ocr_receipt_easyocr(image_path):
    # Предобработка изображения
    preprocessed_image = preprocess_image(image_path)
    # Инициализация EasyOCR reader
    reader = easyocr.Reader(['ru', 'en'], gpu=True)
    # Распознавание текста
    result = reader.readtext(preprocessed_image, min_size=10, contrast_ths=0.05, adjust_contrast=0.5,
                             text_threshold=0.5, detail=0, decoder='beamsearch', paragraph=True)
    return result


image_path = 'src_images/IMG_4275.JPG'
recognized_text = ocr_receipt_easyocr(image_path)

# Вывод распознанного текста
print(recognized_text)
