#!/usr/bin/env python
# coding: utf-8

from utils_file import *

def color_check(fname):
    img = cv2.imread(fname)
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_range = np.array([100,100,100])
        upper_range = np.array([120,255,255])
        mask = cv2.inRange(hsv, lower_range, upper_range)
        res = cv2.bitwise_and(img,img,mask = mask) 
        
        if np.max(res)>0:
            text = 'valid'
            #print(fname.split('/')[-1],'Enable')
        else:
            text = 'invalid'
            #print(fname.split('/')[-1],'Disable')
        return text
    except:
        pass

def detect(source,img_size = 416):
    
    with torch.no_grad():
        flag = True
        for i in range(0,2):
            if i==0:
                weights='box_cut/converted.weights'
                names='box_cut/object.names'
                cfg='box_cut/yolov3.cfg'

            if i==1:
                source="temp/"+source.split('/')[-1]
                weights='enable_box/converted.weights'
                names='enable_box/object.names'
                cfg='enable_box/yolov3.cfg'
                flag = False

            # Initialize
            device = torch.device('cpu')

            # Initialize model
            model = Darknet(cfg, img_size)
            _ = load_darknet_weights(model, weights)
            # Eval mode
            model.to(device).eval()
            dataset = LoadImages(source, img_size=img_size, half=False)

            # Get names and colors
            names = load_classes(names)
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
            # Run inference
            t0 = time.time()
            for path, img, im0s, vid_cap in dataset:
                # Get detections
                img = torch.from_numpy(img).to(device)
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = model(img)[0]
                pred = non_max_suppression(pred,conf_thres=thres,nms_thres=nms_thres)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0 = path, '', im0s
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        for *xyxy, conf, cls in det:
                            pass
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            x1=c1[0]
            y1=c1[1]
            x2=c2[0]
            y2=c2[1]

            if not os.path.isdir('temp/'):
                os.mkdir('temp/')

            if flag:
                crop_img = im0[y1:y2, x1:x2]
                path = "temp/"+source.split('/')[-1]
                cv2.imwrite(path,crop_img)
            else:
                crop_img = im0[y1:y2, x2:x2+abs(x1-x2)+10]
                path = "temp/_"+source.split('/')[-1]
                cv2.imwrite(path,crop_img)
                text = color_check(path)
                #print(text)
                return text

# for i in glob.glob('testing/*')[0:1]:
#     detect(i)