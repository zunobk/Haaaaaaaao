import cv2
import utils
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, make_response
import os
import json
import re
import shutil
from PIL import Image
import numpy as np

from object_tracking.segment.segmentation import SamControler

app = Flask(__name__)


class Server():
    def __init__(self, create_table_query, UPLOAD_FOLDER):
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        self.app.secret_key = os.urandom(24)  
        self.db = utils.Database()  
        self.db.execute_query(create_table_query)  
        self.detectron = utils.Detector()  
        self.segmentation = None
        self.video_faces = []  
        self.video_faces = []
        self.frame_points = []
        self.frame_labels = []
        self.segany_frame = None

        self.app.route('/', methods=['GET'])(self.index)
        self.app.route('/close_modal', methods=['POST'])(self.close_modal)
        self.app.route('/inpaint', methods=['POST'])(self.inpaint)
        self.app.route('/reset', methods=['POST'])(self.reset_image)
        self.app.route('/point_seg', methods=['POST'])(self.frame_load)
        self.app.route('/re_inpaint', methods=['POST'])(self.re_inpaint)
        self.app.route('/submit_face', methods=['POST'])(self.submit_face)
        self.app.route('/upload_video', methods=['POST'])(self.upload_video)
        self.app.route('/object_reset', methods=['POST'])(self.object_reset)
        self.app.route('/objecttracking', methods=['POST'])(self.object_tracking)
        self.app.route('/save_coordinates', methods=['POST'])(self.save_coordinates)

    def close_modal(self):
        del self.segmentation
        return '', 204
    
    def reset_image(self):
        data = request.get_json()
        video_name = data.get('video_name')
        video_state = self.db.select_video_state(video_name)  
        people_list = utils.read_first_line_from_coordinates(video_state=video_state) 
        if os.path.exists(video_state['ouput_path']):
            os.remove(video_state['ouput_path'])
            os.remove(video_state['inpaint_ouput_path'])

        if os.path.exists(video_state['seg_ouput_path']):
            os.remove(video_state['seg_ouput_path'])
            os.remove(video_state['segweb_output_path'])

        for filename in os.listdir(video_state['origin_images_path']):
            src_file_path = os.path.join(video_state['origin_images_path'], filename)
            dest_file_path = os.path.join(video_state['tracking_images_path'], filename)
            dest2_file_path = os.path.join(video_state['inpainting_images_path'], filename)
            if os.path.isfile(src_file_path):
                shutil.copy(src_file_path, dest_file_path)
                shutil.copy(src_file_path, dest2_file_path)
                
        self.detectron.segmentation_image(video_state=video_state, people_list=people_list) 
        best_faces_dir = os.path.join('static/videos', video_state['video_name'], 'best_faces').replace("\\", "/")
        if not os.path.exists(best_faces_dir):
            print("Error: best_faces_dir does not exist.")
        best_face_files = [f for f in os.listdir(best_faces_dir) if f.endswith('.png') or f.endswith('.jpg')]
        best_faces = []
        for index, face in enumerate(best_face_files):
            best_faces.append({
                'image': face,
                'disabled': False  # 나중에 선택된 얼굴을 관리하기 위해 사용될 수 있음
            })
        # best_faces를 세션에 저장
        session['best_faces'] = best_faces                
        return jsonify({'status': 'success','file_name': video_state['video_name'], "best_faces" : best_faces})

    def index(self):
        return render_template('upload.html')

    def upload_video(self):
        file = request.files['video']
        if file:
            filename = file.filename
            select_users_query = "SELECT videoname FROM states;"
            users = self.db.execute_read_query(select_users_query)
            
            result = None
            for item in users:
                if item[0] == filename:
                    result = item[0]
                    break

            if result is None:
                file.save(os.path.join(self.app.config['UPLOAD_FOLDER'], filename))  
                video_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                video_state = utils.get_video(video_path)  
                facial_analysis = utils.InsightFace()  
                
                facial_analysis.process_frames(video_state)
                
                video_state_json = json.dumps(video_state)
                insert_user_query = f"""
                INSERT INTO
                states (videoname, videostate)
                VALUES
                ('{filename}', '{video_state_json}');
                """
                self.db.execute_query(insert_user_query)
            
            video_state = self.db.select_video_state(filename)
            people_list = utils.read_first_line_from_coordinates(video_state=video_state)
                
            origin_video = video_state["origin_images_path"]
            video_state["origin_images_path"] = video_state["inpainting_images_path"]
            No_name = self.detectron.segmentation_image(video_state=video_state, people_list=people_list)
            video_state["origin_images_path"] = origin_video
            
            best_faces_dir = os.path.join('static/videos', filename, 'best_faces').replace("\\", "/")
            if not os.path.exists(best_faces_dir):
                print("Error: best_faces_dir does not exist.")
            best_face_files = [f for f in os.listdir(best_faces_dir) if f.endswith('.png') or f.endswith('.jpg')]

            best_faces = []
            for index, face in enumerate(best_face_files):
                best_faces.append({
                    'image': face,
                    'disabled': False  # 나중에 선택된 얼굴을 관리하기 위해 사용될 수 있음
                })

            # best_faces를 세션에 저장
            session['best_faces'] = best_faces
            print(1)
            return jsonify({'status': 'success','file_name': file.filename, "best_faces" : best_faces, 'no_name' : No_name})

    def submit_face(self):        
        data = request.get_json()  # JSON 데이터를 가져옵니다

        video_name = data.get('video_name')
        selected_face = int(data.get('selected_face'))
        print(selected_face)

        video_state = self.db.select_video_state(video_name)
        person_state = utils.select_face(video_state=video_state, face_id=selected_face)

        # 얼굴 추적 및 마스크 처리
        for i in range(len(person_state)):
            utils.clear_gpu_memory()
            tracking_model = utils.Model_load()
            tracking_model.cutie.clear_memory()

            self.detectron.mask_image(video_state=video_state, person_state=person_state[i])
            mask = utils.change_mask(video_state=video_state, person_state=person_state[i])

            if os.path.exists(video_state['ouput_path']):
                origin_frame = video_state["origin_images_path"]
                video_state["origin_images_path"] = video_state["inpainting_images_path"]

                if i < len(person_state) - 1:
                    tracking_model.tracking_person(video_state=video_state, mask=mask, person_state=(
                        person_state[i]['frame_number'], person_state[i + 1]['frame_number']))
                else:
                    tracking_model.tracking_person(video_state=video_state, mask=mask,
                                                   person_state=(
                                                       person_state[i]['frame_number'], video_state['total_frame']))
                video_state["origin_images_path"] = origin_frame
            else:
                if i < len(person_state) - 1:
                    tracking_model.tracking_person(video_state=video_state, mask=mask, person_state=(
                        person_state[i]['frame_number'], person_state[i + 1]['frame_number']))
                else:
                    tracking_model.tracking_person(video_state=video_state, mask=mask,
                                                   person_state=(
                                                       person_state[i]['frame_number'], video_state['total_frame']))

        for i in reversed(range(len(person_state))):
            utils.clear_gpu_memory()
            tracking_model = utils.Model_load()
            tracking_model.cutie.clear_memory()

            self.detectron.mask_image(video_state=video_state, person_state=person_state[i])
            mask = utils.change_mask(video_state=video_state, person_state=person_state[i])

            if os.path.exists(video_state['ouput_path']):
                origin_frame = video_state["origin_images_path"]
                video_state["origin_images_path"] = video_state["inpainting_images_path"]

                if i > 0:
                    tracking_model.rev_tracking_person(video_state=video_state, mask=mask, person_state=(
                        person_state[i]['frame_number'], person_state[i - 1]['frame_number']))
                else:
                    tracking_model.rev_tracking_person(video_state=video_state, mask=mask,
                                                       person_state=(person_state[i]['frame_number'], 0))
                video_state["origin_images_path"] = origin_frame
            else:
                if i > 0:
                    tracking_model.rev_tracking_person(video_state=video_state, mask=mask, person_state=(
                        person_state[i]['frame_number'], person_state[i - 1]['frame_number']))
                else:
                    tracking_model.rev_tracking_person(video_state=video_state, mask=mask,
                                                       person_state=(person_state[i]['frame_number'], 0))

        utils.generate_video_from_frames(video_state=video_state)
        segweb_output_path = re.sub(r'\\', r'/', video_state["segweb_output_path"])
        original_video_path = os.path.join('./static/videos/uploads/', video_name).replace("\\", "/")
        print(original_video_path)

        response = {'seg_video_path' : segweb_output_path, 'video_name':video_name, 'original_video_path': original_video_path}
        return jsonify(response)

    def inpaint(self):
        data = request.get_json()  # JSON 데이터를 가져옵니다

        video_name = data.get('video_name')
        video_state = self.db.select_video_state(video_name)
        

        
        # 

        if os.path.exists(video_state['ouput_path']):
            utils.main(origin_images_path=video_state['inpainting_images_path'],
                       mask_images_path=video_state['mask_images_path'], ouput_path=video_state['ouput_path'],
                       inpaint_ouput_path=video_state['inpaint_ouput_path'],
                       inpainting_images_path=video_state["inpainting_images_path"])
        else:
            utils.main(origin_images_path=video_state['origin_images_path'],
                       mask_images_path=video_state['mask_images_path'], ouput_path=video_state['ouput_path'],
                       inpaint_ouput_path=video_state['inpaint_ouput_path'],
                       inpainting_images_path=video_state["inpainting_images_path"])

        best_faces = utils.search_video_bestfaces(video_name, self.video_faces)

        # 원본 비디오 경로 생성
        original_video_path = os.path.join('static/videos/uploads', video_name).replace("\\", "/")
        response = {'video_name' : video_name, 'original_video_path':original_video_path,  'inpaint_ouput_path':video_state["inpaint_ouput_path"]}
        return jsonify(response)

    def re_inpaint(self):
        data = request.get_json()  # JSON 데이터를 가져옵니다
        print(data)
        video_name = data.get('video_name')
        video_state = self.db.select_video_state(video_name)

        video_state["origin_images_path"] = video_state["inpainting_images_path"]
        people_list = utils.read_first_line_from_coordinates(video_state=video_state)
        No_name = self.detectron.segmentation_image(video_state=video_state, people_list=people_list)

        for filename in os.listdir(video_state['inpainting_images_path']):
            src_file_path = os.path.join(video_state['inpainting_images_path'], filename)
            dest_file_path = os.path.join(video_state['tracking_images_path'], filename)
            if os.path.isfile(src_file_path):
                shutil.copy(src_file_path, dest_file_path)

        response = {'video_name' : video_name, 'no_name' : No_name}
        return jsonify(response)

    def object_tracking(self):
        del self.segmentation
        data = request.get_json()
        video_name = data.get('video_name')
        frame = data.get('frame')
        video_state = self.db.select_video_state(video_name)
        image = Image.open(video_state['video_frame_path'] + '/seg_anything/masking_image.jpg').convert('L')
        image_np = np.array(image)
        binary_mask = image_np > 0
        print(frame, video_name, video_name, video_state['total_frame'])

        for filename in os.listdir(video_state['inpainting_images_path']):
            src_file_path = os.path.join(video_state['inpainting_images_path'], filename)
            dest_file_path = os.path.join(video_state['tracking_images_path'], filename)
            if os.path.isfile(src_file_path):
                shutil.copy(src_file_path, dest_file_path)

        video_state['origin_images_path'] = video_state['inpainting_images_path']
        utils.clear_gpu_memory()
        tracking_model = utils.Model_load()
        tracking_model.cutie.clear_memory()
        tracking_model.tracking_person(video_state=video_state, mask=binary_mask,
                                       person_state=(int(frame), int(video_state['total_frame'])))

        utils.generate_video_from_frames(video_state=video_state)
        segweb_output_path = re.sub(r'\\', r'/', video_state["segweb_output_path"])
        original_video_path = os.path.join('./static/videos/uploads/', video_name).replace("\\", "/")

        response = {'seg_video_path' : segweb_output_path, 'video_name':video_name, 'original_video_path': original_video_path}
        return jsonify(response)

    def save_coordinates(self):
        utils.clear_gpu_memory()
        data = request.get_json()  # 요청 본문에서 JSON 데이터를 가져옴
        x = data.get('x')
        y = data.get('y')
        new_frame = data.get('frame')
        if not new_frame == self.segany_frame:
            self.frame_points = []
            self.frame_labels = []
            self.segany_frame = new_frame

        
        video_name = data.get('video_name')
        video_state = self.db.select_video_state(video_name)

        if self.segany_frame == new_frame:
            self.frame_points.append((int(x), int(y)))
            self.frame_labels.append(1)
            self.segmentation.sam_controler.reset_image()
            image = Image.open(video_state['inpainting_images_path'] + f"/{self.segany_frame}.jpg")
            image_array = np.array(image)
            mask, painted_image = self.segmentation.create_mask(image=image_array, points=np.array(self.frame_points),
                                                           labels=np.array(self.frame_labels))
        else:
            self.frame_points.append((int(x), int(y)))
            self.frame_labels.append(1)

            self.segmentation.sam_controler.reset_image()
            image = Image.open(video_state['inpainting_images_path'] + f"/{self.segany_frame}.jpg")
            image_array = np.array(image)
            mask, painted_image = self.segmentation.create_mask(image=image_array, points=np.array(self.frame_points),
                                                           labels=np.array(self.frame_labels))

        os.makedirs(video_state['video_frame_path'] + "/seg_anything", exist_ok=True)
        painted_image.save(video_state['video_frame_path'] + "/seg_anything" + "/segmentation_image.jpg")
        binary_mask = mask.astype(np.uint8) * 255
        rgb_mask = cv2.merge([binary_mask, binary_mask, binary_mask])
        cv2.imwrite(video_state['video_frame_path'] + "/seg_anything" + "/masking_image.jpg", rgb_mask)
        
        response = {'new_image_path': video_state['video_frame_path'] + "/seg_anything" + "/segmentation_image.jpg"}
        utils.clear_gpu_memory()
        return jsonify(response)

    def frame_load(self):
        data = request.get_json()  # JSON 데이터를 가져옵니다
        self.segmentation = SamControler()
        video_name = data.get('video_name')
        video_state = self.db.select_video_state(video_name)
        return jsonify({'total_frames': video_state['total_frame'],
                        'frame_path': video_state['inpainting_images_path']})

    def object_reset(self):
        data = request.get_json()  # JSON 데이터를 가져옵니다

        video_name = data.get('video_name')
        video_state = self.db.select_video_state(video_name)
        self.frame_points = []
        self.frame_labels = []
        self.segany_frame = None
        
        os.remove(f"./static/videos/{video_name}/seg_anything/masking_image.jpg")
        os.remove(f"./static/videos/{video_name}/seg_anything/segmentation_image.jpg")
        
        return jsonify({'frame_Path' : video_state['inpainting_images_path']})

        
if __name__ == "__main__":
    UPLOAD_FOLDER = 'static/videos/uploads'
    create_table_query = """CREATE TABLE IF NOT EXISTS states(videoname varchar(256) NOT NULL PRIMARY KEY,videostate LONGTEXT NOT NULL);"""

    server = Server(create_table_query=create_table_query, UPLOAD_FOLDER=UPLOAD_FOLDER)
    server.app.run(host="0.0.0.0", port=5001, debug=True)
