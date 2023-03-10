"""
人脸考勤
人脸注册：将人脸特征存进feature.csv
人脸识别：将检测的人脸特征与CSV中人脸特征作比较，如果比中的把考勤记录写入 attendance.csv
"""

# 导入包
import cv2
import numpy as np
import dlib
import time
import csv
from argparse import ArgumentParser

#face detection
def faceRegister(label_id=1,name='Targe User',count=5,interval=2):
    """
    label_id:人脸ID
    Name:人脸姓名
    count:采集数量
    interval：采集间隔时间
    """
    # 检测人脸
    # 获取68个关键点
    # 获取特征描述符


    cap = cv2.VideoCapture(0)

    #width & height for Video
    width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height =  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #hog face detector: 
    hog_face_detector = dlib.get_frontal_face_detector()
    #shape detector：
    shape_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')
    #feature descriptor. 
    face_descriptor_extractor = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')

    #start_time
    start_time = time.time()

    #collect_count
    collect_count = 0

    # CSV Writer
    f = open('./data/feature.csv','a',newline="")
    csv_writer = csv.writer(f)

    while True:
        ret,frame = cap.read()

        #if the image is too large 
        frame = cv2.resize(frame,(width//2,height//2))

        # mirroring 
        frame =  cv2.flip(frame,1)

        # gray_img
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


        detections = hog_face_detector(frame,1)

        # loop through face
        for face in detections:
            
            # face coordinates
            l,t,r,b =  face.left(),face.top(),face.right(),face.bottom()

            # points on face
            points = shape_detector(frame,face)

            for point in points.parts():
                cv2.circle(frame,(point.x,point.y),2,(0,255,0),-1)

            # face_rectangle 
            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)


            # collect:

            if collect_count < count:
                # time 
                now = time.time()
                # time interval
                if now -start_time > interval:

                    # face_descriptor
                    face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame,points)

                    # list-Fy
                    face_descriptor =  [f for f in face_descriptor]

                    # 写入CSV 文件
                    line = [label_id,name,face_descriptor]

                    csv_writer.writerow(line)


                    collect_count +=1

                    start_time = now

                    print("collect_count:{collect_count}".format(collect_count= collect_count))


                else:
                    pass

            else:
                # 
                print('done!!!')
                return 



        # 显示画面

        cv2.imshow('Face attendance',frame)

        # 退出条件
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    f.close()
    cap.release()
    cv2.destroyAllWindows()    


# 获取并组装CSV文件中特征
def getFeatureList():
    # 构造列表
    label_list = []
    name_list = []
    feature_list = None

    with open('./data/feature.csv','r') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:

            label_id = line[0]
            name = line[1]

            label_list.append(label_id)
            name_list.append(name)
            # string 转为list
            face_descriptor = eval(line[2])
            # 
            face_descriptor = np.asarray(face_descriptor,dtype=np.float64)
            face_descriptor = np.reshape(face_descriptor,(1,-1))

            if feature_list is None:
                feature_list =  face_descriptor
            else:
                feature_list = np.concatenate((feature_list,face_descriptor),axis=0)
    return label_list,name_list,feature_list

#face classification
"""
1.face_feature descriptor
2.comparison between newly captured face & data base
3.make predictions based on ID & NAME
"""

def faceRecognizer(threshold = 0.5):

    cap = cv2.VideoCapture(0)

    # 获取长宽
    width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height =  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 构造人脸检测器
    hog_face_detector = dlib.get_frontal_face_detector()
    # 关键点检测器
    shape_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')

    # feature descriptor for new face
    face_descriptor_extractor = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')

    # read features from new face 
    label_list,name_list,feature_list = getFeatureList()

    while True:
        ret,frame = cap.read()

        # 缩放
        frame = cv2.resize(frame,(width//2,height//2))

        # 镜像
        frame =  cv2.flip(frame,1)

       
        # 检测人脸
        detections = hog_face_detector(frame,1)

        #
        recog_record = {}

        # 遍历人脸
        for face in detections:
            
            # 人脸框坐标
            l,t,r,b =  face.left(),face.top(),face.right(),face.bottom()

            # 获取人脸关键点
            points = shape_detector(frame,face)


            # 矩形人脸框
            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)

            # 获取特征描述符
            face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame,points)

            # 转为列表
            face_descriptor =  [f for f in face_descriptor]

            # 计算与库的距离
            face_descriptor = np.asarray(face_descriptor,dtype=np.float64)


            distances = np.linalg.norm((face_descriptor-feature_list),axis=1)
            # 最短距离索引
            min_index = np.argmin(distances)
            # 最短距离
            min_distance = distances[min_index]

            if min_distance < threshold:
                

                predict_id = label_list[min_index]
                predict_name = name_list[min_index]
                

                if predict_name in recog_record:
                    pass
                else:
                    recog_record[predict_name] = time.time()
                    print(predict_name) 
            else:
                print('Unknow')

        
        cv2.imshow('Face attendance',frame)

        # 退出条件
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    

    cap.release()
    cv2.destroyAllWindows()    


# variables
parser = ArgumentParser()
parser.add_argument("--mode", type=str, default='reg',
                    help="")
parser.add_argument("--id", type=int, default=1,
                    help="")   

parser.add_argument("--name", type=str, default='Target User',
                   help="")                       

parser.add_argument("--interval", type=int, default=2,
                    help="")    

parser.add_argument("--count", type=int, default=5,
                    help="")        
                    
parser.add_argument("--threshold", type=float, default=0.5,
                    help="")                                                                                         

args = parser.parse_args()



mode = args.mode

if mode == 'regis':
    faceRegister(label_id=args.id,name=args.name,interval=args.interval,count=args.count)

if  mode == 'recog':
    faceRecognizer(threshold=args.threshold)