let globalFileName = '';
let insertFilename = '';
let sliderInitialized = false;
let clickListenerAdded = false;
let currentFrame = 0;
const videoUploadPreview = document.getElementById('videoUploadPreview');
const videoList = document.getElementById('videoList');
const videoFileInput = document.getElementById('videoFile');




function previewVideo() {
    var file = document.getElementById('videoFile').files[0];
    var preview = document.getElementById('videoUploadPreview');
    var reader = new FileReader();

    reader.onloadend = function () {
        preview.src = reader.result;
        preview.style.display = 'block';
    }

    if (file) {
        reader.readAsDataURL(file);
        globalFileName = file.name
        document.getElementById('hiddenFileName').value = globalFileName;
        insertFilename = globalFileName
        sendFileName(file.name)

        videoInput.value = '';
    } else {
        preview.src = "";
        preview.style.display = 'none';
    }
}

function showLoadingScreen(message = '처리 중...') {
    const loadingScreen = document.getElementById('loading-screen');
    loadingScreen.querySelector('p').textContent = message;
    loadingScreen.classList.remove('hidden');
}

function hideLoadingScreen() {
    document.getElementById('loading-screen').classList.add('hidden');
}

function sendFileName(fileName) {
    fetch('/edited_video_load', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fileName: fileName })
    })
    .then(response => response.json())
    .then(data => {
        videoList.innerHTML = ''; // Clear existing list 

        data.inpainting_video_files.forEach(video => {
            const listItem = document.createElement('li');
            listItem.textContent = video;
            listItem.className = 'video-item';

            listItem.addEventListener('click', () => {
                const videoUrl = `/static/videos/${fileName}/web_inpainting_videos/${video}`;
                videoUploadPreview.src = videoUrl;
                videoUploadPreview.style.display = 'block'; // Show the video preview
                insertFilename = video
            });

            videoList.appendChild(listItem);
        });
    })
    .catch(error => {
        console.error('Error sending file name:', error);
    });
}

document.getElementById('videoUploadForm').addEventListener('submit', function (event) {
    event.preventDefault(); // 기본 폼 제출 동작을 막습니다.
    document.getElementById('descriptionInput').disabled = true;

    showLoadingScreen('얼굴 인식 중...');  // 로딩 화면 표시

    const formData = new FormData(this);
    formData.append('globalFileName', globalFileName);
    formData.append('insertFilename', insertFilename);
    const descriptionInputValue = document.getElementById('descriptionInput').value;
    formData.append('user_edit_name', descriptionInputValue);
    fetch('/upload_video', { // 서버의 파일 업로드 엔드포인트를 여기에 입력합니다.
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(result => {
            hideLoadingScreen();

            if (result.status === 'success') {
                if (!result || Object.keys(result).length === 0) {
                    document.getElementById('descriptionInput').disabled = false;
                    throw new Error('Empty response received from server');
                }

                console.log('서버 응답:', result); 
                populateFaces(result.best_faces, result.file_name);
                document.getElementById('tracking_button').classList.remove('hidden');
                document.getElementById('segmentaion_image').classList.remove('hidden');
                document.getElementById('hiddenFileName').value = result.file_name;
                toggleFaceOptions(result.no_name);
                document.getElementById('imageSelect').scrollIntoView({ behavior: 'smooth' });
            } else {
                alert('Video upload failed.');
            }
        })
        .catch(error => {
            hideLoadingScreen();
            console.error('Error:', error);
            alert('An error occurred while uploading the video.');
            document.getElementById('descriptionInput').disabled = false;
        });
});

function populateFaces(faces, videoName) {
    const facesContainer = document.getElementById('faces-container');
    facesContainer.innerHTML = '';  // 기존 내용을 지웁니다.
    
    faces.forEach((face, index) => {
        const faceItem = document.createElement('div');
        faceItem.className = 'image-option face-item';

        const radioInput = document.createElement('input');
        radioInput.type = 'radio';
        radioInput.id = `face${index + 1}`;
        radioInput.name = 'selected_face';
        radioInput.value = index + 1;
        radioInput.addEventListener('change', () => showSelectedFace(videoName, index + 1));

        const label = document.createElement('label');
        label.htmlFor = radioInput.id;

        const img = document.createElement('img');
        img.src = `/static/videos/${videoName}/best_faces/${face.image}`;
        img.alt = 'Best Face';

        label.appendChild(img);
        faceItem.appendChild(radioInput);
        faceItem.appendChild(label);

        facesContainer.appendChild(faceItem);
    });
}

function showSelectedFace(videoName, faceIndex, isSegImage = true) {
    const selectedImage = document.getElementById('selectedImage');
    let imgSrc;
    if (isSegImage) {
        imgSrc = `/static/videos/${videoName}/seg_images/ID_${faceIndex}_Seg.jpg`;
    } else {
        // 얼굴 이미지 URL 생성
        const selectedFace = document.querySelector(`input[name="imageSelect"][value="${faceIndex}"]`);
        if (selectedFace) {
            imgSrc = selectedFace.nextElementSibling.querySelector('img').src;
        }
    }

    if (imgSrc) {
        selectedImage.src = imgSrc;
        document.getElementById('hiddenSelectedFace').value = faceIndex;
    }
}

function submitForm() {
    const selectedFaceId = document.querySelector('input[name="selected_face"]:checked')?.value;
    if (selectedFaceId) {
        showLoadingScreen('객체 추적 중...');  // 로딩 화면 표시
        const selectedFaceIds = JSON.parse(sessionStorage.getItem('selectedFaceIds')) || [];
        selectedFaceIds.push(selectedFaceId);
        sessionStorage.setItem('selectedFaceIds', JSON.stringify(selectedFaceIds));

        fetch('/submit_face', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ selected_face: selectedFaceId, video_name :  document.getElementById('hiddenFileName').value})  // 선택된 얼굴 ID를 JSON 형식으로 전송
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
                
            }
            return response.json();
        })
        .then(data => {
            console.log('서버 응답:', data);  
            const originalVideoSource = document.getElementById('originalVideoSource');
            const segmentationVideoSource = document.getElementById('segmentationVideoSource');

            originalVideoSource.src = `${data.original_video_path}?t=${new Date().getTime()}`;
            segmentationVideoSource.src = `${data.seg_video_path}?t=${new Date().getTime()}`;   

            document.getElementById('originalVideoPlayer').load();
            document.getElementById('segmentaionVideoPlayer').load();
            document.getElementById('segentation_video').scrollIntoView({ behavior: 'smooth' });

            hideLoadingScreen();
        })
        .catch(error => {
            console.error('서버 요청 오류:', error);
            hideLoadingScreen();  // 오류 발생 시에도 로딩 화면 숨기기
            alert('객체 추적 처리 중 오류가 발생했습니다.');
            document.getElementById('descriptionInput').disabled = false;
        });
        
    } else {
        alert("Please select a face before submitting.");
    }
}

function openModal() {
    document.getElementById('myModal').style.display = 'block';
}

function closeModal() {
    document.getElementById('myModal').style.display = 'none';
    fetch('/close_modal', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
}

function fetchConfigAndOpenModal() {
    showLoadingScreen('창 띄우는 중...');

    fetch('/point_seg', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ video_name: document.querySelector('input[name="file_name"]').value })
    })
        .then(response => response.json())
        .then(config => {
            const slider = document.getElementById('frame-slider');
            const image = document.getElementById('frame-image');
            const currentFrameDisplay = document.getElementById('current-frame');

            // 서버에서 받은 총 프레임 수와 이미지 경로 설정
            const maxFrames = config.total_frames;
            const framePath = config.frame_path;

            // 슬라이더 설정
            slider.max = maxFrames - 1;
            slider.value = currentFrame;
            currentFrameDisplay.textContent = slider.value;

            function updateImage(frameNumber) {
                image.src = `${framePath}/${frameNumber}.jpg`;
                currentFrame = frameNumber;
            }

            // 이미지 클릭 이벤트 추가
            image.addEventListener('click', function (event) {
                showLoadingScreen('객체 인식 중...');

                const rect = image.getBoundingClientRect();
                const x = event.clientX - rect.left; // 클릭된 위치의 X 좌표
                const y = event.clientY - rect.top;  // 클릭된 위치의 Y 좌표
            
                // 이미지의 실제 크기와 표시 크기 비율을 계산
                const actualWidth = image.naturalWidth;
                const actualHeight = image.naturalHeight;
                const displayWidth = rect.width;
                const displayHeight = rect.height;
            
                // 클릭된 좌표를 실제 이미지 크기 비율에 맞게 조정
                const actualX = x * (actualWidth / displayWidth);
                const actualY = y * (actualHeight / displayHeight);
            
                fetch('/save_coordinates', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ x: actualX, y: actualY, frame: currentFrame, video_name: globalFileName })
                })
                    .then(response => response.json())  // 서버에서 JSON 응답 받기
                    .then(data => {
                        // 서버에서 받은 새로운 이미지 경로를 사용하여 모달 업데이트
                        if (data.new_image_path) {
                            image.src = `${data.new_image_path}?t=${new Date().getTime()}`;  // 쿼리 파라미터로 시간 추가
                            document.getElementById('click-coordinates').textContent = `좌표: (${Math.round(actualX)}, ${Math.round(actualY)})`;
                        } else {
                            console.error('서버 응답에서 이미지 경로를 찾을 수 없습니다.');
                        }
                    })
                    .catch(error => {
                        console.error('서버 전송 오류:', error);
                    })
                    .finally(() => {
                        hideLoadingScreen();  // 객체 인식이 완료되면 로딩 화면 숨기기
                    });
            });

            slider.addEventListener('input', function () {
                const frameNumber = slider.value;
                updateImage(frameNumber);
                currentFrameDisplay.textContent = frameNumber;
            });

            // 마지막 프레임으로 이미지 로드
            updateImage(currentFrame);
            openModal();
            hideLoadingScreen();
        })
        .catch(error => {
            console.error('Error fetching config:', error);
            hideLoadingScreen();
        });
}

function closeModal() {
    const modal = document.getElementById('myModal');
    modal.style.display = 'none';  // 모달을 닫습니다.
}

function handleYes() {
    // 모달 닫기
    closeModal();
    showLoadingScreen('객체 추적 중...');  // 로딩 화면 표시

    // 비디오 이름과 현재 프레임 정보를 가져옵니다.
    const videoName = document.querySelector('input[name="file_name"]').value;
    const currentFrame = document.getElementById('current-frame').textContent;

    // 서버에 POST 요청을 보냅니다.
    fetch('/objecttracking', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ video_name: videoName, frame: currentFrame,
             user_edit_name : document.getElementById('descriptionInput').value
        })
    })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                console.error('Object tracking 요청 실패:', response.statusText);
            }
        })
        .then(data => {
            const originalVideoSource = document.getElementById('originalVideoSource');
            const segmentationVideoSource = document.getElementById('segmentationVideoSource');

            originalVideoSource.src = `${data.original_video_path}?t=${new Date().getTime()}`;
            segmentationVideoSource.src = `${data.seg_video_path}?t=${new Date().getTime()}`;   

            document.getElementById('originalVideoPlayer').load();
            document.getElementById('segmentaionVideoPlayer').load();
            document.getElementById('segentation_video').scrollIntoView({ behavior: 'smooth' });

            hideLoadingScreen();
        })
        .catch(error => {
            hideLoadingScreen();
            console.error('오류 발생:', error);
        });
}

function handleNo() {
    closeModal();
}

function reset_frame() {
    fetch('/object_reset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ video_name: document.querySelector('input[name="file_name"]').value })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            const image = document.getElementById('frame-image');
            const framePath = data.frame_Path

            function updateImage(frameNumber) {
                image.src = `${framePath}/${frameNumber}.jpg`;
                currentFrame = frameNumber;
            }
            updateImage(currentFrame);
        })
        .catch(error => {
            console.error('Object tracking reset 오류:', error);
        });
}

document.getElementById('inpaint-form').addEventListener('submit', function(event) {
event.preventDefault(); 

    const fileName = document.getElementById('hiddenFileName').value;
    
    if (!fileName) {
        alert('파일 이름이 설정되지 않았습니다.');
        return;
    }

    showLoadingScreen('인페인팅 처리 중...'); // 로딩 화면 표시

    fetch('/inpaint', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
         body: JSON.stringify({ video_name: fileName, user_edit_name : document.getElementById('descriptionInput').value })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('서버 응답:', data);
        const originalVideoSource2 = document.getElementById('originalVideoSource2');
        const inpaintVideoSource = document.getElementById('inpaintVideoSource');

        originalVideoSource2.src = `${data.original_video_path}?t=${new Date().getTime()}`;
        inpaintVideoSource.src = `${data.inpaint_ouput_path}?t=${new Date().getTime()}`;   

        document.getElementById('originalVideoPlayer2').load();
        document.getElementById('InpaintingVideoPlayer').load();
        document.getElementById('video2').scrollIntoView({ behavior: 'smooth' });

        hideLoadingScreen(); // 로딩 화면 숨기기
    })
    .catch(error => {
        console.error('서버 요청 오류:', error);
        hideLoadingScreen(); // 오류 발생 시에도 로딩 화면 숨기기
        alert('인페인팅 처리 중 오류가 발생했습니다.');
    });
});

document.getElementById('re-inpaint-form').addEventListener('submit', function(event) {
    event.preventDefault();  // 폼 제출을 방지
    const fileName = document.getElementById('hiddenFileName').value;
    console.log('File Name:', fileName);
        
        if (!fileName) {
            alert('파일 이름이 설정되지 않았습니다.');
            return;
        }

        showLoadingScreen('동영상 처리 중...'); // 로딩 화면 표시

        fetch('/re_inpaint', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ video_name: fileName })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            
            console.log('서버 응답:', data);

            toggleFaceOptions(data.no_name);

            document.getElementById('imageSelect').scrollIntoView({ behavior: 'smooth' });
            hideLoadingScreen();
        })
        .catch(error => {
            console.error('서버 요청 오류:', error);
            hideLoadingScreen(); // 오류 발생 시에도 로딩 화면 숨기기
            alert('인페인팅 처리 중 오류가 발생했습니다.');
        });
});

function toggleFaceOptions(noNameIds) {
    const faceOptions = document.querySelectorAll('input[name="selected_face"]');
    
    faceOptions.forEach(option => {
        if (noNameIds.includes(option.value)) {
            option.disabled = false;  // no_name에 포함된 경우 활성화
        } else {
            option.disabled = true;   // no_name에 포함되지 않은 경우 비활성화
        }
    });
}

document.getElementById('reset-form').addEventListener('click', function(event) {
    event.preventDefault();  // 폼 제출을 방지
    const fileName = document.getElementById('hiddenFileName').value;

    showLoadingScreen('되돌리는 중...'); // 로딩 화면 표시

    fetch('/reset', {
        method: 'POST',  // POST 요청
        headers: {
            'Content-Type': 'application/json',  // JSON 형식의 데이터를 전송
        },
        body: JSON.stringify({ video_name: fileName })  // 빈 객체를 전송하거나 필요한 데이터를 포함시킬 수 있음
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('서버 응답:', data);
        if (data.status === 'success') {
            document.getElementById('descriptionInput').disabled = false;
            populateFaces(data.best_faces, data.file_name);
            document.getElementById('tracking_button').classList.remove('hidden');
            document.getElementById('segmentaion_image').classList.remove('hidden');
            document.getElementById('hiddenFileName').value = data.file_name;
            document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
            hideLoadingScreen(); // 오류 발생 시에도 로딩 화면 숨기기
        } else {
            alert('Video upload failed.');
            hideLoadingScreen(); // 오류 발생 시에도 로딩 화면 숨기기
        }
    })
    .catch(error => {
        console.error('서버 요청 오류:', error);
        hideLoadingScreen(); // 오류 발생 시에도 로딩 화면 숨기기
    });
});










// 페이지가 로드될 때 다음 섹션으로 자동 이동
window.addEventListener('DOMContentLoaded', function () {
    const nextSectionId = "{{ next_section_id }}";
    if (nextSectionId) {
        scrollToSection(nextSectionId);
    }
});

// 스크롤을 특정 섹션으로 이동시키는 함수
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
    }
}

// Glitch Timeline Function
var $filter = document.querySelector('.svg-sprite'),
    $turb = $filter.querySelector('#filter feTurbulence'),
    turbVal = {val: 0.000001},
    turbValX = {val: 0.000001};

var glitchTimeline = function () {
    var timeline = new TimelineMax({
        repeat: -1,
        repeatDelay: 2,
        paused: true,
        onUpdate: function () {
            $turb.setAttribute('baseFrequency', turbVal.val + ' ' + turbValX.val);
        }
    });

    timeline
        .to(turbValX, 0.1, {val: 0.5})
        .to(turbVal, 0.1, {val: 0.02});
    timeline
        .set(turbValX, {val: 0.000001})
        .set(turbVal, {val: 0.000001});
    timeline
        .to(turbValX, 0.2, {val: 0.4}, 0.4)
        .to(turbVal, 0.2, {val: 0.002}, 0.4);
    timeline
        .set(turbValX, {val: 0.000001})
        .set(turbVal, {val: 0.000001});

    return {
        start: function () {
            timeline.play(0);
        },
        stop: function () {
            timeline.pause();
        }
    };
};

var btnGlitch = new glitchTimeline();

document.querySelectorAll('.btn').forEach(function (btn) {
    btn.addEventListener('mouseenter', function () {
        this.classList.add('btn-glitch-active');
        btnGlitch.start();
    });
    btn.addEventListener('mouseleave', function () {
        if (this.classList.contains('btn-glitch-active')) {
            this.classList.remove('btn-glitch-active');
            btnGlitch.stop();
        }
    });
});