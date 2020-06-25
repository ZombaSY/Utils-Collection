import os

image_dir = 'A:/Users/SSY/Desktop/image_calibration_dataset/dataset'
transfer_destination_input = 'A:/Users/SSY/Desktop/image_calibration_dataset/A'
transfer_destination_output = 'A:/Users/SSY/Desktop/image_calibration_dataset/B'
image_list = os.listdir(image_dir)

for idx, image_name in enumerate(image_list):
    image_path = os.path.join(image_dir, image_name)

    f_read = open(image_path, 'rb')

    if image_path[-5] == '1':
        f_write_input = open(os.path.join(transfer_destination_input, image_name), 'wb')
        f_write_input.write(f_read.read())
        f_write_input.close()
    elif image_path[-5] == '2':
        f_write_output = open(os.path.join(transfer_destination_output, image_name), 'wb')
        f_write_output.write(f_read.read())
        f_write_output.close()
    else:
        Exception('Unexpected Labelling Type!!!')

    f_read.close()


