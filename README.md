# nEmil id validation

# Objective: Classify email_id as valid or invalid email_id.

This repository is an implementation of YOLO v3 to detect the bounding box and validate a email_id. 
There are two models trained on top of each other. One is to detect the bonding box of the compose box 
and other is to detect the bounding boxes of the option box that need to be classified and which is responsible for 
identifying email_id as valid or invalid.

# Steps to classifiy an Email_id is valid or invalid:

1. We trained a custom object detection model to detect:
    (1) compose box and saved this box in a folder.
    (2) Cut the two unique icon from the compose box and based on the 2nd icon position we get crop the next two icons and saved it in a folder.
2. Convert this icon image in hsv colour format.
3. Identify the range of blue colour and pass it in the lower and upper range.
4. Masking the values that are not belonging to the lower and upper range by black colour and value of black colour is 0.
5. Apply the threshold that if the value is greate than 0 then it is valid otherwise invalid.

# Input:- 
The image below contains the email id in the email box that need to be validate. This is the input to the model and the validation of email_id is done on it.


![Screenshot](https://github.com/nka218/email_id_validation/blob/main/backup/1_d.png)



- If  the below options are in blue then the email id is valid else it is invalid.

![Screenshot](https://github.com/nka218/email_id_validation/blob/main/backup/valid_invalid.png)




# Directory Structure:

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
		
# Links for model weights:-
Link1 :- https://drive.google.com/file/d/1TAubicDFBfjAY8ZBTI4VSLyfeHyTUJPF/view?usp=sharing
Link2 :- https://drive.google.com/file/d/1aek4o-mRTHvGyEkBlOC1skyk3RPvnCo7/view?usp=sharing
		
# How to use API:

1. Clone the mail_box_identification API from Gitlab repsitiory.
2. Extract the cloned zip file.
3. Run pip install -r requirements.txt.
4. Download model weights from Link1  and put it into box_cut folder.( box_cut/converted.weights )
5. Download model weights from Link2 and put it into enable_boxfolder.( enable_box/converted.weights )
6. Run python 3 new_mail.py
7. Use radio button the visualize the required images.
