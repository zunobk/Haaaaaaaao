import utils
from flask import Flask, render_template, request, redirect, url_for
import os
import json
import re

class Server():
    def __init__(self, create_table_query, UPLOAD_FOLDER):
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        self.db = utils.Database()
        self.db.execute_query(create_table_query)
        
        self.detectron = utils.Detector()
        self.video_faces = []


        self.app.route('/', methods=['GET'])(self.index)
        self.app.route('/upload_video', methods=['POST'])(self.upload_video)
        self.app.route('/submit_face', methods=['POST'])(self.submit_face)
        self.app.route('/inpaint', methods=['POST'])(self.inpaint)
        self.app.route('/re_inpaint', methods=['POST'])(self.re_inpaint)

    def index(self):
        return render_template('upload.html')

    def upload_video(self):
        if 'video' not in request.files:    # 'video' 필드가 요청 파일에 없는 경우
            return 'No file part'

        file = request.files['video']       # 'video' 파일 객체 가져오기
        if file.filename == '':             # 파일 이름이 비어 있는지 확인
            return 'No selected file'       
        if file:                            # 파일이 존재하는 경우
            filename = file.filename        # 파일 이름 가져오기
            file.save(os.path.join(self.app.config['UPLOAD_FOLDER'], filename))      # 파일을 업로드 폴더에 저장

            video_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)    # 업로드된 파일의 전체 경로를 생성
            video_state = utils.get_video(video_path)

            select_users_query = "SELECT videoname FROM states;"
            users = self.db.execute_read_query(select_users_query)

            result = None
            for item in users:
                if item[0] == filename:
                    result = item[0]
                    break
                
            if result == None:
                facial_analysis = utils.InsightFace()
                facial_analysis.process_frames(video_state)

                people_list = utils.read_first_line_from_coordinates(video_state= video_state)

                self.detectron.segmentation_image(video_state=video_state, people_list=people_list)

                # video_state를 JSON 문자열로 변환
                video_state_json = json.dumps(video_state)
                insert_user_query = f"""
                INSERT INTO
                states (videoname, videostate)
                VALUES
                ('{filename}', '{video_state_json}');
                """
                self.db.execute_query(insert_user_query)

            best_faces_dir = os.path.join('static/videos', filename, 'best_faces').replace("\\", "/") 
            best_faces_dir = f"./{best_faces_dir}" 
            if not os.path.exists(best_faces_dir):
                print("Error: best_faces_dir does not exist.") 
            best_face = [f for f in os.listdir(best_faces_dir) if f.endswith('.png') or f.endswith('.jpg')]

            best_face = {
            "video_name" : filename,
            "best_faces" : best_face
            }
            self.video_faces.append(best_face)
            best_faces = utils.search_video_bestfaces(filename, self.video_faces) 

            return render_template('upload.html', best_faces=best_faces, video_name=filename)

    def submit_face(self):
        selected_face = int(request.form['selected_face'])
        video_name = request.form['video_name']

        video_state = self.db.select_video_state(video_name)
        person_state = utils.select_face(video_state=video_state, face_id=selected_face)

        for i in range(len(person_state)):
            utils.clear_gpu_memory()
            tracking_model = utils.Model_load()
            tracking_model.cutie.clear_memory()
            
            
            self.detectron.mask_image(video_state=video_state, person_state=person_state[i])
            mask = utils.change_mask(video_state=video_state, person_state=person_state[i])

            if os.path.exists(video_state['ouput_path']):
                origin_frame=video_state["origin_images_path"]
                video_state["origin_images_path"]=video_state["inpainting_images_path"]

                if i < len(person_state) - 1:    
                    tracking_model.tracking_person(video_state=video_state, mask=mask, person_state=(
                                                    person_state[i]['frame_number'], person_state[i + 1]['frame_number']))
                else:
                    tracking_model.tracking_person(video_state=video_state, mask=mask,
                                                    person_state=(person_state[i]['frame_number'], video_state['total_frame']))
                video_state["origin_images_path"]=origin_frame
            else:                
                if i < len(person_state) - 1:
                    tracking_model.tracking_person(video_state=video_state, mask=mask, person_state=(
                                                    person_state[i]['frame_number'], person_state[i + 1]['frame_number']))
                else:
                    tracking_model.tracking_person(video_state=video_state, mask=mask,
                                                    person_state=(person_state[i]['frame_number'], video_state['total_frame']))
                    
        utils.generate_video_from_frames(video_state=video_state)
        segweb_output_path = re.sub(r'\\', r'/', video_state["segweb_output_path"])

        best_faces = utils.search_video_bestfaces(video_name, self.video_faces)

        return render_template('upload.html', best_faces=best_faces, video_name=video_name, video_path=segweb_output_path)

    def inpaint(self):
        video_name = request.form['video_name']
        video_state = self.db.select_video_state(video_name)

        if os.path.exists(video_state['ouput_path']):
            utils.main(origin_images_path=video_state['inpainting_images_path'], mask_images_path=video_state['mask_images_path'], ouput_path=video_state['ouput_path'], 
                 inpaint_ouput_path=video_state['inpaint_ouput_path'], inpainting_images_path=video_state["inpainting_images_path"])
        else:
            utils.main(origin_images_path=video_state['origin_images_path'], mask_images_path=video_state['mask_images_path'], ouput_path=video_state['ouput_path'], 
                 inpaint_ouput_path=video_state['inpaint_ouput_path'], inpainting_images_path=video_state["inpainting_images_path"])

        best_faces = utils.search_video_bestfaces(video_name, self.video_faces)
        print(best_faces)
        return render_template('upload.html', best_faces=best_faces, video_name=video_name,
                               video_path=video_state["segweb_output_path"],
                               inp_video_path=video_state["inpaint_ouput_path"])

    def re_inpaint(self):
        video_name = request.form['video_name']
        best_faces = utils.search_video_bestfaces(video_name, self.video_faces)
        video_state = self.db.select_video_state(video_name)

        video_state["origin_images_path"]=video_state["inpainting_images_path"]
        people_list = utils.read_first_line_from_coordinates(video_state= video_state)
        self.detectron.segmentation_image(video_state=video_state, people_list=people_list)

        return render_template('upload.html', best_faces=best_faces, video_name=video_name)

if __name__ == "__main__":
    UPLOAD_FOLDER = 'static/videos/uploads'
    create_table_query = """CREATE TABLE IF NOT EXISTS states(videoname varchar(256) NOT NULL PRIMARY KEY,videostate LONGTEXT NOT NULL);"""
    
    server =Server(create_table_query=create_table_query, UPLOAD_FOLDER=UPLOAD_FOLDER)
    server.app.run(host="0.0.0.0", port=5001, debug=True)

