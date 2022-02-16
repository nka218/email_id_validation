# email_id_validationEmail id validation

Objective: Classify email_id as valid or invalid email_id.

This repository is an implementation of YOLO v3 to detect the bounding box and validate a email_id. 
There are two models trained on top of each other. One is to detect the bonding box of the compose box 
and other is to detect the bounding boxes of the option box that need to be classified and which is responsible for 
identifying email_id as valid or invalid.

Steps to classifiy an image in enable or disable:

1. We trained a custom object detection model to detect:
    (1) compose box and saved this box in a folder.
    (2) Cut the two unique icon from the compose box and based on the 2nd icon position we get crop the next two icons and saved it in a folder.
2. Convert this icon image in hsv colour format.
3. Identify the range of blue colour and pass it in the lower and upper range.
4. Masking the values that are not belonging to the lower and upper range by black colour and value of black colour is 0.
5. Apply the threshold that if the value is greate than 0 then it is enable otherwise disable.

Original image:- The image below contains the email id in the email box that need to be validate.






- If  the below options are in blue then the email id is valid else it is invalid.

Valid Email_id:-						Invalid Email_id:-






Directory Structure:

	|- root directory
		|- assets
		|- models.py 
		|- utils_file.py
		|- backup
		|- new_mail.py
		|- temp
		|- box_cut
		|- enable_box 
		|- lawplus.log
		|- output 
		|- Video_demo.mp4
		|- cut_out  jupyter_pipeline.py 
		|-  __pycache
		|- requirements.txt
		|- Readme.md
		
How to use API:

1. Clone the mail_box_identification API from Gitlab repsitiory.
2. Extract the cloned zip file.
3. Run pip install -r requirements.txt.
4. Run python 3 new_mail.py
5. Use radio button the visualize the required images.
