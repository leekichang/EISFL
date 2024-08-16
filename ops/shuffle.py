__all__ = ['ImageShuffle']

class ImageShuffle:
    def __init__(self, patch_size, patch_shuffle, pixel_shuffle):
        self.patch_size    = patch_size
        self.patch_shuffle = patch_shuffle
        self.pixel_shuffle = pixel_shuffle

    def shuffle(self, images):
        if self.patch_shuffle==None and self.pixel_shuffle==None: return images
        b, c, h, w = images.shape
        ph, pw = self.patch_size

        # Reshape the images into patches
        # images = b * c * h * w -> patches = b * c * nh * w * ph -> patches = b * c * nh * nw * ph * pw
        patches = images.unfold(2, ph, ph).unfold(3, pw, pw)
        # patches = b * nh * nw * c * ph * pw -> patches = b * (nh*nw) * c * (ph*pw)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, -1, c, ph*pw)
        
        # Shuffle patches (nh * nw)
        if self.patch_shuffle != None:
            patches = patches[:, self.patch_shuffle]
        # Shuffle patches (ph * pw)
        if self.pixel_shuffle != None:
            patches = patches[:, :, :, self.pixel_shuffle]
        
        # Reshape the shuffled patches back to the original images shape
        # patches = b * nh * nw * c * ph * pw
        images = patches.view(b, h // ph, w // pw, c, ph, pw)
        # patches = b * c * nh * ph * nw * pw -> patches = b * c * h * w
        images = images.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, h, w)
        return images

    def __call__(self, image):
        # Shuffle patches and pixels in each patch
        return self.shuffle(image)