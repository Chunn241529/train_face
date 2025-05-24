@echo off
chcp 65001

echo Kiểm tra môi trường ảo... 🔍 

REM Kiểm tra xem môi trường ảo đã tồn tại chưa
if exist ".venv" (
    echo Môi trường ảo đã tồn tại. ✅
) else (
    echo Đang tạo môi trường ảo... ⚙️ 
    python -m venv .venv

    REM Kiểm tra việc tạo môi trường ảo có thành công không
    if not exist ".venv" (
        echo ❌ Lỗi khi tạo môi trường ảo!
        exit /b 1
    )
)

REM Kích hoạt môi trường ảo
echo Kích hoạt môi trường ảo... 🔑 
call .venv\Scripts\activate

REM Nâng cấp pip nếu cần
echo Nâng cấp pip... ⬆️
python -m pip install --upgrade pip

REM Cài đặt thư viện
echo Đang cài đặt thư viện... 📦 
pip install --upgrade flet markdown requests
pip install --upgrade duckduckgo_search
pip install --upgrade beautifulsoup4 pillow numpy
pip install opencv-python
pip install matplotlib
pip install scikit-learn


REM Ghi thư viện đã cài vào requirements.txt (bằng UTF-8 có dấu)
pip freeze > requirements.txt

REM Cài đặt lại từ requirements.txt để xác nhận
echo Double check... 📦 
pip install -r requirements.txt

echo ✅ Hoàn tất!
pause
