class ImageProcessingException(Exception):
    def __init__(self, message):
        self.message = message


class CirclesNotFoundException(ImageProcessingException):
    def __init__(self, *args):
        super().__init__(*args)


class PupilOutsideIrisException(ImageProcessingException):
    def __init__(self, *args):
        super().__init__(*args)
