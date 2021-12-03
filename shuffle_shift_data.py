import os
import random
import shutil

import click

@click.command()
@click.option("--input_path", help="Input image source")
@click.option("--output_path", help="Input image sink")
@click.option("--shift_amount", help="Number of images to shift")
def shuffle_and_shift(input_path, output_path, shift_amount):
    files_list = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                files_list.append(os.path.join(root, file))
    file_count = len(files_list)
    print("Total files in source folder: ",file_count)

    # print files_list
    filesToCopy = random.sample(files_list, int(shift_amount))
    destPath = output_path

    # if destination dir does not exists, create it
    if os.path.isdir(destPath) == False:
            os.makedirs(destPath)

    # iteraate over all random files and move them
    for file in filesToCopy:
        shutil.move(file, destPath)

if __name__ == "__main__":
    shuffle_and_shift()
