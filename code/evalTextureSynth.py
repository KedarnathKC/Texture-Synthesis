import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy import ndimage
import time

np.random.seed(1)

def synthRandomPatch(img, tileSize, numTiles, outSize):
    outImage = np.ones((outSize,outSize,3),dtype=np.uint8)
    r,c=0,0
    while(r<outSize):
        c=0
        while(c<outSize):
            randR=np.random.randint(0,img.shape[0]-tileSize+1)
            randC=np.random.randint(0,img.shape[1]-tileSize+1)
            outImage[r:r+tileSize,c:c+tileSize,:]=img[randR:randR+tileSize,randC:randC+tileSize,:3]
            c+=tileSize
        r+=tileSize
    plt.imshow(outImage[:,:,:3])
    plt.show()
    return outImage

def getNoofNghs(pixelList,mask,winSize):
    pixels = dict()
    maskOnes = np.ones((winSize,winSize))
    for pixel in pixelList:
        x,y = pixel
        #Doing this to get the transform of the pixel indexes from dilation to binImg, since binImg is padded with winSize//2 no of zeros on each side
        x,y=x+winSize//2,y+winSize//2
        mul = mask[x-winSize//2:x+winSize//2 + 1, y-winSize//2:y+winSize//2 + 1]
        pixels[(x,y)]=np.sum(np.multiply(mul,maskOnes))
    return pixels
    
def getPatchFromImg(Img,winSize):
    patches = list()
    for i in range(Img.shape[0]-winSize+1):
        for j in range(Img.shape[1]-winSize+1):
            patches.append(np.ndarray.flatten(Img[i:i+winSize,j:j+winSize]))
    return np.array(patches)

def SSD(patches,validMask,validImg):
    diff = patches-validImg
    return np.sum(np.square(diff*validMask),axis=1)
    
def synthEfrosLeung(img, winSize, outSize):
    start = time.time()
    h,w = outSize+winSize-1,outSize+winSize-1

    paddedImg =np.pad(img[:,:,0],winSize//2)
    outImg = np.zeros((h,w),dtype=np.uint8)
    binImg = np.zeros(outImg.shape)
    finalImg = np.zeros((outSize,outSize,3),dtype=np.uint8)

    # Random patch for the initialization
    randR=np.random.randint(0,img.shape[0]-winSize+1)
    randC=np.random.randint(0,img.shape[1]-winSize+1)

    outImg[h//2-winSize//2 :h//2 + winSize//2 +1 ,w//2-winSize//2 :w//2 + winSize//2 +1] = img[randR:randR+winSize,randC:randC+winSize,0]
    binImg[outImg>1]=1
    patches = getPatchFromImg(paddedImg,winSize)
    cnt=0
    while(cnt<outSize*outSize-winSize*winSize):
        dilate =  ndimage.binary_dilation(binImg[winSize//2:outSize+winSize//2,winSize//2:outSize+winSize//2],structure=np.ones((winSize,winSize))).astype(outImg.dtype) - binImg[winSize//2:outSize+winSize//2,winSize//2:outSize+winSize//2]
        pixelList = np.transpose(np.nonzero(dilate>0))
        pixels=getNoofNghs(pixelList,binImg,winSize)
        sortedPixelList = sorted(pixels.items(), key=lambda item: item[1], reverse=True)
        
        for pixel in sortedPixelList:
            cnt+=1
            x,y = pixel[0]
            validMask = np.ndarray.flatten(binImg[x-winSize//2:x+winSize//2+1 , y-winSize//2:y+winSize//2+1])[np.newaxis,:]
            validMask = np.repeat(validMask, repeats=patches.shape[0], axis=0)
            validImg = np.repeat(np.ndarray.flatten(outImg[x-winSize//2:x+winSize//2+1 , y-winSize//2:y+winSize//2+1])[np.newaxis,:],repeats=patches.shape[0],axis=0)
            ssd=SSD(patches,validMask,validImg)
            ssdMin = np.min(ssd)
            bestMatches = np.where(ssd<=ssdMin*1.1)[0]
            bestMatch = np.random.choice(bestMatches)
            bestMatchX,bestMatchY = bestMatch//img.shape[1],bestMatch%img.shape[1]
            outImg[x,y]=img[bestMatchX,bestMatchY,0]
            binImg[outImg>0]=1
            finalImg[:,:,0] = outImg[winSize//2:outSize+winSize//2,winSize//2:outSize+winSize//2]
            finalImg[:,:,1] = outImg[winSize//2:outSize+winSize//2,winSize//2:outSize+winSize//2]
            finalImg[:,:,2] = outImg[winSize//2:outSize+winSize//2,winSize//2:outSize+winSize//2]
    end = time.time()
    print("Time: ",(end-start)* 10**3)
    # plt.imshow((finalImg * 255).astype(np.uint8))
    plt.imshow(finalImg)
    plt.show()
    return finalImg



# Load images
img = io.imread('./data/texture/D20.png')
# img = io.imread('./data/texture/Texture2.bmp')
# img = io.imread('./data/texture/english.jpg')


# Random patches
tileSize = 40 # specify block sizes
numTiles = 5
outSize = numTiles * tileSize # calculate output image size

# implement the following, save the random-patch output and record run-times
im_patch = synthRandomPatch(img, tileSize, numTiles, outSize)


# Non-parametric Texture Synthesis using Efros & Leung algorithm  
winsize = 15 # specify window size (5, 7, 11, 15)
outSize = 70 # specify size of the output image to be synthesized (square for simplicity)

# implement the following, save the synthesized image and record the run-times
im_synth = synthEfrosLeung(img, winsize, outSize)



