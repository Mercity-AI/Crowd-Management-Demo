# Crowd Management Demo

A real-time crowd management system using YOLOv8 for person detection and tracking. This system can process multiple video feeds to track and count people in crowds. 

Built and managed by [Mercity AI](https://mercity.ai)!

## Demo
![Crowd Tracking Demo 1](demo/demo1.gif)
![Crowd Tracking Demo 2](demo/demo2.gif)

## Features

- Real-time person detection and tracking using YOLOv8
- Multiple video processing support
- Person counting and ID tracking
- Visual bounding boxes and tracking IDs
- Support for various video formats (mp4, avi, mov, mkv)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mercity-AI/Crowd-Management-Demo.git
cd Crowd-Management-Demo
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your videos in the `videos` directory
2. Run the processing script:
```bash
python process_videos.py
```
3. Check the `results` directory for processed videos

## Project Structure

```
.
├── videos/          # Input videos directory
├── results/         # Processed videos with tracking
├── demo/           # Demo GIFs
├── process_videos.py
├── requirements.txt
└── README.md
```

## Output

The processed videos include:
- Green bounding boxes around detected people
- Person ID numbers for tracking
- Total person count in the top-left corner

## Requirements

- Python 3.8 or higher
- OpenCV
- PyTorch
- Ultralytics (YOLOv8)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 for the object detection model
- ByteTrack for the tracking algorithm 
