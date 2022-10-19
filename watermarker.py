import os
import cv2
import numpy as np
import argparse
import moviepy.editor as mp
import logging
import sys

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


def video_watermark(video_path : str, watermark_path : str, output_path : str):
    video = mp.VideoFileClip(video_path)

    logo = (mp.ImageClip(watermark_path)
            .set_duration(video.duration)
            #.resize(height=video.h//4) # if you need to resize...
            .resize(height=video.h//4) # if you need to resize...
            #.margin(right=8, bottom=8, opacity=0) # (optional) logo-border padding
            .set_pos(("left","top")))

    final = mp.CompositeVideoClip([video, logo])
    final.write_videofile(output_path)

def main():

    if getattr(sys, 'frozen', False):
        start_dir = os.path.dirname(sys.executable)
    else:
        start_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(start_dir) # Move to script folder

    parser = argparse.ArgumentParser(description="Add a watermark to all images in a folder.")
    parser.add_argument('--watermark', metavar="watermark", type=str, help="Path to the watermark image", default="watermark.png")
    parser.add_argument('--inputfolder', metavar="inputfolder", type=str, help="Path to the folder containing the images", default="input")
    parser.add_argument('--outputfolder', metavar="out_folder", type=str, help="Path to output folder", default="output")
    args = parser.parse_args()

    print(args)

    watermark_path = args.watermark
    input_folder = args.inputfolder
    output_folder = args.outputfolder

    image_extensions = [".png", ".tiff", ".jpg", ".jpeg"]
    video_extensions = [".mp4"]

    files = os.listdir(input_folder)
    watermark_img = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)

    logging.debug(files)

    for folder, subs, files in os.walk(input_folder):
        for file in files:
            if file == watermark_path:
                continue
            local_folder = os.sep.join(os.path.normpath(folder).split(os.sep)[1:])
            output_file_path = os.path.join(output_folder, local_folder, file)
            logging.debug(local_folder, file, output_file_path)
            if os.path.exists(output_file_path):
                continue
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            ext = os.path.splitext(file)[-1].lower()
            logging.debug(ext)
            if ext in image_extensions:
                source_img = cv2.imread(os.path.join(folder, file))
                watermarked = apply_watermark(source_img, watermark_img)
                cv2.imwrite(output_file_path, watermarked)
            elif ext in video_extensions:
                video_watermark(os.path.join(folder, file), watermark_path, output_file_path)
            else:
                print("no matching extension")


if __name__ == "__main__":
    main()
