import os
import shutil

def create_mini(path_to_files, mini_perc):
    path_to_mini = "./mini_codenet/data/"
    for prob in os.listdir(path_to_files):
        # create problem folder
        os.mkdir(path_to_mini + prob)
        for lang in os.listdir(path_to_files + prob + "/"):
            # create language folder
            os.mkdir(path_to_mini + prob + "/" + lang)
            ten_perc = int(len(os.listdir(path_to_files + prob + "/" + lang))*mini_perc)
            idx = 0
            for file_name in os.listdir(path_to_files + prob + "/" + lang):
                # copy file over
                src = path_to_files + prob + "/" + lang + "/" + file_name
                dst = path_to_mini + prob + "/" + lang + "/" + file_name
                shutil.copyfile(src, dst)

                idx += 1
                if idx == ten_perc:
                    break
            

if __name__ == '__main__':
    mini_percent = 0.1 # percentage of the original dataset that mini should consist of | MODIFY-ABLE
    if mini_percent <= 1.0:
        create_mini('./Project_CodeNet/data/', mini_percent)
        print('Finished creating mini_codenet')
    else:
        print('ERROR: MAKE SURE THAT mini_percent IS <= 1.0')
