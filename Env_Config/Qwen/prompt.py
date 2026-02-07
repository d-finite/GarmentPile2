import numpy as np

def Prompt (
    resolution : list,
    description : str,
    labels : np.ndarray
):
    prompt = (f"Given one {resolution[0]}x{resolution[1]} input image and the following instructions, proceed as follows:\n"
            "1. Target object selection:\n"
            "You are performing a clothing-grasping task.\n"
            "The image shows the current clothing scene with several numbered markers.\n"
            "Because descriptions of clothing items (e.g., shirts, pants) and colors (e.g., white, black) in the instructions are subjective, you should allow for slight discrepancies when comparing them to the image.\n"
            "Identify in the image the object that best matches the instruction. If such an object exists and is suitable for grasping (i.e. it is sufficiently visible or not overly covered by other garments), select it as the target object.\n"
            "If the target is not visible in the image, then consider importance and safety and choose the most cost-effective object as the target.\n"
            "By 'cost-effectiveness', we mean: which garment would you remove to most likely reveal the target? Obviously, any piece that covers a large area is more likely to be obstructing the target object.\n"
            "2. Grasp‐point selection:\n"
            "I will provide you with coordinates of numbered markers in the image.\n"
            "Different numerical markers denote different regions. Each region is either a distinct object or a part of the same object with markedly different characteristics.\n"
            "After selecting the target object, please choose from the provided coordinates the point that is closest to and most representative of your chosen target object.\n"
            "The point you provide must lie on the target object. Grasping the garment at this point with the gripper should be the most convenient, the most efficient, and the most natural\n"
            "In general, a point near the center of the garment is easiest to grasp, while points at the edges are harder to grip.\n"
            "3. Output should be in the following format:\n"
            "Selected Target Object: [object: color and name]\n"
            "Point for grasp: (x,y), selection numbered marker\n"
            "4. Instructions are as follows:\n"
            f"{description}\n"
            "5. The coordinates corresponding to each numbered marker:\n"
            )
    for idx, (x, y, r, g, b) in enumerate (labels, start=1):
        prompt += f"  {idx}. ({int(x)},{int(y)}), RGB=({int(r)},{int(g)},{int(b)})\n"
    return prompt


def Prompt_closed_scene_all_garments (
    resolution : list,
    labels : np.ndarray
):
    prompt = (f"Given one {resolution[0]}x{resolution[1]} input image and the following instructions, proceed as follows:\n"
            "1. Target object selection:\n"
            "You are performing a clothing-grasping task.\n"
            "The image shows the current clothing scene with several numbered markers.\n"
            "All clothes are placed in the clothes basket. You need to find the clothes basket first and distinguish it from the background.\n"
            "Please select a piece of clothing that is easiest to grasp, lift and remove from the basket as the target object."
            "(by 'easiest' we mean, for example, garments with a large exposed area or those not entangled with other clothes).\n"
            "2. Grasp‐point selection:\n"
            "I will provide you with coordinates of numbered markers in the image.\n"
            "Different numerical markers denote different regions. Each region is either a distinct object or a part of the same object with markedly different characteristics.\n"
            "After selecting the target object, please choose from the provided coordinates the point that is closest to and most representative of your chosen target object.\n"
            "The point you provide must lie on the target object. Grasping the garment at this point with the gripper should be the most convenient, the most efficient, and the most natural\n"
            "In general, a point near the center of the garment is easiest to grasp, while points at the edges are harder to grip.\n"
            "3. Output should be in the following format:\n"
            "Selected Target Object: [object: color and name]\n"
            "Point for grasp: (x,y), selection numbered marker\n"
            "4. The coordinates corresponding to each numbered marker:\n"
            )
    for idx, (x, y, r, g, b) in enumerate (labels, start=1):
        prompt += f"  {idx}. ({int(x)},{int(y)})\n"
    return prompt


def Prompt_open_scene_all_garments (
    resolution : list,
    labels : np.ndarray
):
    prompt = (f"Given one {resolution[0]}x{resolution[1]} input image and the following instructions, proceed as follows:\n"
            "1. Target object selection:\n"
            "You are performing a clothing-grasping task.\n"
            "The image shows the current clothing scene with several numbered markers.\n"
            "All clothes are placed on the table. You need to find the table first and distinguish it from the background.\n"
            "Please select a piece of clothing that is easiest to grasp, lift and remove from the basket as the target object."
            "(by 'easiest' we mean, for example, garments with a large exposed area or those not entangled with other clothes).\n"
            "2. Grasp‐point selection:\n"
            "I will provide you with coordinates of numbered markers in the image.\n"
            "Different numerical markers denote different regions. Each region is either a distinct object or a part of the same object with markedly different characteristics.\n"
            "After selecting the target object, please choose from the provided coordinates the point that is closest to and most representative of your chosen target object.\n"
            "The point you provide must lie on the target object. Grasping the garment at this point with the gripper should be the most convenient, the most efficient, and the most natural\n"
            "In general, a point near the center of the garment is easiest to grasp, while points at the edges are harder to grip.\n"
            "3. Output should be in the following format:\n"
            "Selected Target Object: [object: color and name]\n"
            "Point for grasp: (x,y), selection numbered marker\n"
            "4. The coordinates corresponding to each numbered marker:\n"
            )
    for idx, (x, y, r, g, b) in enumerate (labels, start=1):
        prompt += f"  {idx}. ({int(x)},{int(y)})\n"
    return prompt


def Prompt_check_if_masks_need_regen (
    resolution : list,
    labels : np.ndarray
):
    prompt = (f"Given two {resolution[0]}x{resolution[1]} input images and the following instructions, proceed as follows:\n"
            "1. Image information description:\n"
            "I have used a tool to segment this image, generating several masks for the clothing pile, with each mask corresponding to one piece of clothing.\n"
            "The first image shows the current clothing pile scene with number mark information of each mask.\n"
            "The second image contains both the number marks and the masks, with the masks rendered in random colors and a certain level of transparency.\n"
            "2. Task requirements:\n"
            "When you interpret the information from each numbered marker’s corresponding mask in the second image,"
            "If you find that salient two or more markedly different garments, for example, they have obviously different color (e.g., a blue garment and a red garment), have been segmented into the same mask, that mask is incorrect.\n"
            "To maximum efficiency, you must report the numbered marks corresponding to only critically incorrect masks!\n"
            "Adjusting mask is a extremely cost action. So please think twice!\n"
            "More mask has choosed, longer adjusting time, higher moving cost! I can't afford too much cost!\n"
            "3. Output should be in the following format:\n"
            "Assuming you need to report 1, 5 and 7, you should return: 1,5,7.\n"
            "Assuming you need to report 2, 3 and 6, you should return: 2,3,6.\n"
            "Specifically, if there are no number marks to report, return: 0.\n"
            "To ensure I can retrieve the result, you must return it on the last line by itself and maintain the same format I have provided!"
            )
    return prompt


def Prompt_closed_scene_specific_garment (
    resolution : list,
    description : str,
    labels : np.ndarray,
    CoT_file_path : str = None
):
    prompt = (f"Given one {resolution[0]}x{resolution[1]} input image and the following instructions, proceed as follows:\n"
            "1. Target Object Selection:\n"
            "You are performing a clothing-grasping task.\n"
            "The image shows the current clothing scene with several numbered markers.\n"
            "All clothes are placed in the clothes basket. You need to find the clothes basket first and distinguish it from the background.\n"
            "Because descriptions of clothing items (e.g., shirts, pants) and colors (e.g., white, black) in the instructions are subjective, you should allow for slight discrepancies when comparing them to the image.\n"
            "Identify in the image the object that best matches the instruction. If such an object exists and is suitable for grasping (i.e. it is sufficiently visible or not overly covered by other garments), select it as the target object.\n"
            "If the target is not visible in the image, then consider importance and safety and choose the most cost-effective object as the target.\n"
            "By 'cost-effectiveness', we mean: which garment would you remove to most likely reveal the target? Obviously, any piece that covers a large area is more likely to be obstructing the target object.\n"
            "2. Grasp‐point selection:\n"
            "I will provide you with coordinates of numbered markers in the image.\n"
            "Different numerical markers denote different regions. Each region is either a distinct object or a part of the same object with markedly different characteristics.\n"
            "After selecting the target object, please choose from the provided coordinates the point that is closest to and most representative of your chosen target object.\n"
            "The point you provide must lie on the target object. Grasping the garment at this point with the gripper should be the most convenient, the most efficient, and the most natural\n"
            "In general, a point near the center of the garment is easiest to grasp, while points at the edges are harder to grip.\n"
            "3. Output should be in the following format:\n"
            "Selected Target Object: [object: color and name]\n"
            "Point for grasp: (x,y), selection numbered marker\n"
            "4. Instructions are as follows:\n"
            f"{description}\n"
            "5. The coordinates corresponding to each numbered marker:\n"
            )
    for idx, (x, y, r, g, b) in enumerate (labels, start=1):
        prompt += f"  {idx}. ({int(x)},{int(y)})\n"
   
    if CoT_file_path != None:
        with open (CoT_file_path, 'r', encoding='utf-8') as file:
            content = file.read ()
            prompt += f"\n{content}"
            print (f"check CoT-prompt: len = {len (content)}")
    return prompt


def Prompt_open_scene_specific_garment (
    resolution : list,
    description : str,
    labels : np.ndarray
):
    prompt = (f"Given one {resolution[0]}x{resolution[1]} input image and the following instructions, proceed as follows:\n"
            "1. Target Object Selection:\n"
            "You are performing a clothing-grasping task.\n"
            "The image shows the current clothing scene with several numbered markers.\n"
            "All clothes are placed on the table. You need to find the table first and distinguish it from the background.\n"
            "Because descriptions of clothing items (e.g., shirts, pants) and colors (e.g., white, black) in the instructions are subjective, you should allow for slight discrepancies when comparing them to the image.\n"
            "Identify in the image the object that best matches the instruction. If such an object exists and is suitable for grasping (i.e. it is sufficiently visible or not overly covered by other garments), select it as the target object.\n"
            "If the target is not visible in the image, then consider importance and safety and choose the most cost-effective object as the target.\n"
            "By 'cost-effectiveness', we mean: which garment would you remove to most likely reveal the target? Obviously, any piece that covers a large area is more likely to be obstructing the target object.\n"
            "2. Grasp‐point selection:\n"
            "I will provide you with coordinates of numbered markers in the image.\n"
            "Different numerical markers denote different regions. Each region is either a distinct object or a part of the same object with markedly different characteristics.\n"
            "After selecting the target object, please choose from the provided coordinates the point that is closest to and most representative of your chosen target object.\n"
            "The point you provide must lie on the target object. Grasping the garment at this point with the gripper should be the most convenient, the most efficient, and the most natural\n"
            "In general, a point near the center of the garment is easiest to grasp, while points at the edges are harder to grip.\n"
            "3. Output should be in the following format:\n"
            "Selected Target Object: [object: color and name]\n"
            "Point for grasp: (x,y), selection numbered marker\n"
            "4. Instructions are as follows:\n"
            f"{description}\n"
            "5. The coordinates corresponding to each numbered marker:\n"
            )
    for idx, (x, y, r, g, b) in enumerate (labels, start=1):
        prompt += f"  {idx}. ({int(x)},{int(y)})\n"
    return prompt


def Prompt_check_if_need_rightarm (
    resolution : list,
    labels : np.ndarray,
    last_idx : int
):
    prompt = (f"Given two {resolution[0]}x{resolution[1]} input images and the following instructions, proceed as follows:\n"
            "1. Task description:\n"
            "You are performing a clothing-grasping task.\n"
            "The first image shows the current clothing scene with each piece of clothing labeled with a numbered marker.\n"
            "The single-arm robot has already lifted the target clothing, which you can clearly notice.\n"
            "The second image shows the initial state of the clothing pile.\n"
            f"The clothing currently being grasped is the one with numbered marker {last_idx + 1} in the second image."
            "Observe whether the patterns of the two garments are similar; you will locate this garment in the first image.\n"
            "If you can understand it: this garment, since it has been lifted, should be relatively long in length.\n"
            "Your task is to determine whether this garment requires assistance from the other robotic arm.\n"
            # "Believe me, you always need assistance because these garments are too big."
            # "For example, when the end of the garment have some part dragging on a table, or if it's still have some part in a basket, it require assistance.\n"
            # "Sometimes, even with a large garment, no assistance is needed if it's picked up from a favorable spot.\n"
            "To ensure efficiency, it's best to request assistance primarily when needed.\n"
            "2. Output should be in the following format:\n"
            "If the garment requires assistance from the other robotic arm, return the coordinates corresponding to its numbered marker in the first image: (x,y)\n"
            "Otherwise, return: (0,0)\n"
            "3. The coordinates corresponding to each numbered marker in the first image:\n"
            )
    for idx, (x, y, r, g, b) in enumerate (labels, start=1):
        prompt += f"  {idx}. ({int(x)},{int(y)})\n"
    return prompt


def Prompt_check_if_multiple_picked_garments (
    resolution : list,
):
    prompt = (f"Given one {resolution[0]}x{resolution[1]} input images, proceed as follows:\n"
            "1. Task description:\n"
            "You have lifted garment from clutter scene.\n"
            "The image shows current scene. You can find there are some garments was lifted.\n"
            "I want to lift exactly one garments rather two or more garments per round.\n"
            "So if you find there are two or more garments was lifted, reporting a warning.\n"
            "My robots are precise. So lifting-one-garment is the almostly all situation!\n"
            "Therefore, you should deal with this task very carefully and concerntrating.\n"
            "And you should report a warning when there are two or more garments absolutely obviously lifted.\n"
            "2. Output should be in the following format:\n"
            "For common place, you should return: (0,0).\n"
            "If you persistently decide to report (you must think twice!), return (1,1).\n"
            )
    return prompt


def GenPrompt (
    resolution : list,
    labels : np.ndarray,
    last_idx : int = -1,
    description : str = "",
    prompt_type : str = "default",
    CoT_type : str = None,
):
    label_fix = ("Note! Some numbered markers may not refer to items of clothing (e.g. Robot, Basket, Table, Background ... )."
                 "This is an error in the previous process."
                 "Please carefully identify and only consider the numbered markers of clothing.")

    if CoT_type == None:
        CoT_file_path = None
    else:
        CoT_file_path = f"Env_Config/Qwen/{CoT_type}.txt"

    if prompt_type == "default":
        return Prompt (resolution, description, labels) + label_fix

    elif prompt_type == "closed_scene_specific_garment":
        return Prompt_closed_scene_specific_garment (resolution, description, labels, CoT_file_path) + label_fix

    elif prompt_type == "closed_scene_all_garments":
        return Prompt_closed_scene_all_garments (resolution, labels) + label_fix

    elif prompt_type == "open_scene_specific_garment":
        return Prompt_open_scene_specific_garment (resolution, description, labels) + label_fix

    elif prompt_type == "open_scene_all_garments":
        return Prompt_open_scene_all_garments (resolution, labels) + label_fix

    elif prompt_type == "check_if_masks_need_regen":
        return Prompt_check_if_masks_need_regen (resolution, labels) + label_fix

    elif prompt_type == "check_if_need_rightarm":
        return Prompt_check_if_need_rightarm (resolution, labels, last_idx) + label_fix
    
    elif prompt_type == "check_if_multiple_picked_garments":
        return Prompt_check_if_multiple_picked_garments (resolution)