# Hackathon 2025

## Team Members
- **Khaled Saleh**  
- **Moetaz Mohamed**  
- **Mario El Shaer**  
- **Tarek Elalfi**  

---

## Project Setup

### 1. Set Up a Virtual Environment
Creating a virtual environment ensures that all dependencies are isolated for this project.

#### **For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### **For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

To deactivate the virtual environment when done:
```bash
deactivate
```

---

### 2. Install Dependencies
Run the following command in the **root directory** to install all required dependencies:

```bash
pip install -r requirements.txt
```

---

### 3. Set Up FFmpeg
FFmpeg is required to run voice commands in this project. Follow the steps below to set it up based on your operating system.

#### **For Windows:**
1. **Download FFmpeg:**  
   [Download FFmpeg for Windows](https://ffmpeg.org/download.html) and extract it to a folder (e.g., `C:\ffmpeg`).

2. **Add FFmpeg to Environment Variables:**
   - Open **System Properties** > **Environment Variables**.
   - Under **System variables**, find and select `Path`, then click **Edit**.
   - Click **New** and add the following path:
     ```
     C:\ffmpeg\bin
     ```
   - Click **OK** to save.

3. **Verify Installation:**
   Open a new terminal or command prompt and run:
   ```bash
   ffmpeg -version
   ```
   If FFmpeg is correctly installed, you'll see version information.

#### **For macOS/Linux:**
1. **Install FFmpeg via Homebrew (Recommended):**
   If you don’t have Homebrew, first install it by running:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   Then install FFmpeg:
   ```bash
   brew install ffmpeg
   ```

2. **Verify Installation:**
   Run the following command in the terminal:
   ```bash
   ffmpeg -version
   ```
   You should see the version details if the installation was successful.

---
