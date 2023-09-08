# Process cam_extri.xml and output the transform.json that neus2 requries
import xml.etree.ElementTree as ET
import numpy as np
import json
import random
from pathlib import Path

intrinsics = {'f': 1.75468979e+03, 'cx': 2.76974005e+00, 'cy': -4.51774316e+00,
                'k1':0.08825813346363495, 'k2': -0.030144145083343458, 'k3': 0.004599722123729709,
                'width': 3840, 'height': 2160}

intrinsic_list = [
    [   intrinsics['f'],
        0.0,
        intrinsics['width']/2 - intrinsics['cx'],
        0.0
    ],
    [
        0.0,
        intrinsics['f'],
        intrinsics['height']/2 - intrinsics['cy'],
        0.0
    ],
    [
        0.0,
        0.0,
        1.0,
        0.0
    ],
    [
        0.0,
        0.0,
        0.0,
        1.0
    ]
]

if __name__ == "__main__":
    output_json_file = Path("../transform.json")
    input_xml_file = Path("../cam_extri.xml")


    neus_json = {"w": intrinsics['width'], 
                 "h": intrinsics['height'],
                 "aabb_scale": 128,
                 "scale": 0.5,
                "offset": [
                    0.5,
                    0.5,
                    0.5
                ],
                "from_na": True,
                "frames":[]
                }


    xml_tree = ET.parse(input_xml_file)
    xml_root = xml_tree.getroot()
    
    # 855 images, too many images, randomly sample some images 
    # image_id range: [1, 856)
    num_samples = 50
    sampled_ids = random.sample(range(1, 856), num_samples)

    for camera in xml_root.iter('camera'):
        
        frame_data = {}

        image_id = int(camera.attrib['id']) + 1 # Used to match the image
        if image_id not in sampled_ids:
            continue

        image_file_path = "images/" + "{:06d}.png".format(image_id)
        
        frame_data["file_path"] = image_file_path
        
        
        # Get camera extrinsic
        transform_str = camera[0].text
        transform_str = camera[0].text
        transform_list = transform_str.split()
        transform_ctow = np.array(transform_list, dtype='float').reshape((4,4))
        extrinsic_matrix = np.linalg.inv(transform_ctow)
        extrinsic_list = extrinsic_matrix.tolist()


        frame_data["transform_matrix"] = extrinsic_list

        frame_data["intrinsic_matrix"] = intrinsic_list

        neus_json["frames"].append(frame_data)
        print("Added data for: " + image_file_path)


    neus_json_object = json.dumps(neus_json, indent=4)
    with open(output_json_file, "w") as outfile:
        outfile.write(neus_json_object)