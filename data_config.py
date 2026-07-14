
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = r'Z:\cj\datasourse\LEVIR-CD-256'
        # elif data_name == 'WHU':
        #     self.label_transform = "norm"
        #     self.root_dir = r'E:\bianhuajiance\WHU'
        # elif data_name == 'WHU-512-100':
        #     self.label_transform = "norm"
        #     self.root_dir = r'E:\bianhuajiance\database\WHU-CD-512-100'
        # elif data_name == 'WHU-512-0':
        #     self.label_transform = "norm"
        #     self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-0'
        # elif data_name == 'WHU-512-10':
        #     self.label_transform = "norm"
        #     self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-10'
        # elif data_name == 'WHU-512-20':
        #     self.label_transform = "norm"
        #     self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-20'
        # elif data_name == 'WHU-512-30':
        #     self.label_transform = "norm"
        #     self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-30'
        # elif data_name == 'WHU-512-30-only':
        #     self.label_transform = "norm"
        #     self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-30-only'
        # elif data_name == 'WHU-512-40':
        #     self.label_transform = "norm"
        #     self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-40'
        # elif data_name == 'WHU-512-40-only':
        #     self.label_transform = "norm"
        #     self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-40-only'
        # elif data_name == 'WHU-512-50':
        #     self.label_transform = "norm"
        #     self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-50'
        # elif data_name == 'WHU-512-50-only':
        #     self.label_transform = "norm"
        #     self.root_dir = r'Z:\cj\datasourse\WHU-CD-512-50-only'
        ###########################################################
        elif data_name == 'LEVIR-256-edge':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\LEVIR-CD\merged_data'
        #############################################################
        elif data_name == 'LEVIR-512-edge':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\LEVIR-CD\512merged_data'
        elif data_name == 'LEVIR-1024-edge':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\LEVIR-CD\1024merged_data'
        elif data_name == 'WHU-256-edge':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\Building change detection dataset\WHU_256'
        #######################################################
        elif data_name == 'WHU-256-edge-7-1-2':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\Building change detection dataset\WHU_256_7_1_2'
        ##########################################################
        elif data_name == 'WHU-512-edge':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\Building change detection dataset\WHU_512'
        elif data_name == 'WHU-1024-edge':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\Building change detection dataset\WHU_1024'
        elif data_name == 'WHU-256-edge-7-2-1-no-delete':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\Building change detection dataset\WHU_256_7_2_1'
        elif data_name == 'test_256':
            self.label_transform = "norm"
            self.root_dir = r'F:\Data\test'
        elif data_name == 'CDCD-256-edge-7-1-2':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\CDCD\CDCD_256_7_1_2'
        ##########################################################
        elif data_name == 'CDCD-repair-7-1-2':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\CDCD\CDCD_repair_712'
        ###############################################################
        elif data_name == 'CDCD-check-7-1-2':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\CDCD\CDCD_check_712'
        elif data_name == 'CDCD-9-1-10':
            self.label_transform = "norm"
            self.root_dir = r'E:\zyh\Data\CDCD\CDCD_test'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self

