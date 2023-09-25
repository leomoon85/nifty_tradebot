import os


class ReadmeGenerator:
    def __init__(self, base_url, project_folder, short_name):
        self.base_url = base_url
        self.project_folder = project_folder
        self.short_name = "NIFTY" 

    def write(self):
        my_file = open(os.path.join(self.project_folder, 'README.md'), "w+")
        my_file.write('![](' + self.base_url + self.project_folder + '/' + self.short_name + '_price.png)\n')
        my_file.write('![](' + self.base_url + self.project_folder + '/' + self.short_name + '_hist.png)\n')
        my_file.write('![](' + self.base_url + self.project_folder + '/' + self.short_name + '_prediction.png)\n')
        my_file.write('![](' + self.base_url + self.project_folder + '/' + 'MSE.png)\n')
        my_file.write('![](' + self.base_url + self.project_folder + '/' + 'loss.png)\n')
