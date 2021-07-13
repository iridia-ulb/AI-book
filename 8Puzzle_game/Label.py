import pygame

class Label :
    def __init__(self, text, x, y) :
        """
        Init a label.

        :param text:    The text for the label, String.
        :param x:       The x position for the label, Int.
        :param y:       The y position for the label, Int.
        """
        self.x = x
        self.y = y
        self.font = pygame.font.Font(None, 40)
        self.originalText = text
        self.text = self.font.render(text, 1, pygame.Color("White"))
        self.size = self.w, self.h = self.text.get_size()
        self.rect = pygame.Rect(self.x, self.y, self.w, self.h)
        self.surface = pygame.Surface(self.size)
        self.surface.blit(self.text, (0, 0))
        self.label_clicked = False
    
    def getSurface(self) :
        """
        Getter for the surface of the label.

        :return: Return the surface of the label, Surface object.
        """
        return self.surface
    
    def getText(self) :
        """
        Getter for the original text of the label.

        :return: Return a String, it is the original text of the label.
        """
        return self.originalText
    
    def isClicked(self) :
        """
        Check if the label has been clicked

        :return: Return a Boolean, it is True if the label has been clicked, otherwise False.
        """
        return self.label_clicked
    
    def clicked(self) :
        """
        The label has been clicked.
        """
        self.label_clicked = True