import os

def get_filepaths(directory,extension="json"):
    filepaths = []
    for root,dirs,files in os.walk(directory):
        for _file in files:
            if _file.endswith(extension):
                filepaths.append(os.path.join(root,_file))
    print("size of filepaths = ",len(filepaths))
    return filepaths

def analysis_label():
    pass

if __name__ == "__main__":
    analysis_label()