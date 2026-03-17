# usfq-mmia
 Repository to store my work for the Master's in Artificial Intelligence at the Universidad San Francisco de Quito (USFQ).

 ## Quickstart

 Install [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) to set up the environment:

 ```bash
curl -Ls https://astral.sh/uv/install.sh | sh
 ```

 Then, create the environment and install the dependencies:
 ```bash
uv sync
 ```
 ## Contents
- `cnns` : Convolutional Neural Networks notebooks.

## Specifics
- To run the crnn notebook without warnings the ffmpeg library must be installed. On Linux, this can be done with:
```bash
sudo apt-get install ffmpeg
```
Or manually without using sudo:
```bash
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
tar xvf ffmpeg-git-amd64-static.tar.xz
cd ffmpeg-*-static
mv ffmpeg ffprobe ~/bin/
export PATH="$HOME/bin:$PATH"
source ~/.bashrc
```