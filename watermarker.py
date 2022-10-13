import os
import cv2
import numpy as np
import argparse


def apply_watermark(source_img : np.ndarray, watermark_img : np.ndarray):
    # Calculate aspect ratio of watermark
    watermark_ratio = watermark_img.shape[0]/watermark_img.shape[1]
    # Resize watermark to take up 1/4 of vertical size of source image while maintaining aspect ratio
    resized_watermark = cv2.resize(watermark_img, (source_img.shape[0]//4, int(source_img.shape[0]//4*watermark_ratio)))
    result = source_img.copy()
    # Calculate weights for pixels from source image vs watermark based on watermark PNG alpha channel data.
    weight_mask = (resized_watermark[:,:,3].astype("float64")/255)[:,:,np.newaxis]
    # Extract watermark sized slice from bottom-right corner of original image
    orig_slice = result[result.shape[0] - resized_watermark.shape[0]:, result.shape[1] - resized_watermark.shape[1]:,:].astype("float64")
    # Calculate new BGR values based on watermark weights
    new_slice = weight_mask*resized_watermark[:,:,:3] + (1-weight_mask)*orig_slice
    # Apply new calculated values to the result
    result[result.shape[0] - resized_watermark.shape[0]:, result.shape[1] - resized_watermark.shape[1]:,:] = new_slice.astype("uint8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Add a watermark to all images in a folder.")
    parser.add_argument('--watermark', metavar="watermark", type=str, help="Path to the watermark image", default="watermark.png")
    parser.add_argument('--imagefolder', metavar="img_folder", type=str, help="Path to the folder containing the images", default=".")
    parser.add_argument('--outputfolder', metavar="out_folder", type=str, help="Path to output folder", default="output")
    args = parser.parse_args()

    print(args)

    watermark_path = args.watermark
    image_folder = args.imagefolder
    output_folder = args.outputfolder

    output_path = os.path.join(output_folder, image_folder)
    os.makedirs(output_path, exist_ok=True)

    image_extensions = [".png", ".tiff", ".jpg", ".jpeg"]

    files = os.listdir(image_folder)
    watermark_img = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)

    print(files)

    for file in files:
        print(file)
        ext = os.path.splitext(file)[-1].lower()
        if file == watermark_path or ext not in image_extensions:
            continue
        source_img = cv2.imread(os.path.join(image_folder, file))
        watermarked = apply_watermark(source_img, watermark_img)
        print(os.path.join(output_folder, image_folder, file))
        cv2.imwrite(os.path.join(output_folder, image_folder, file), watermarked)


if __name__ == "__main__":
    main()
