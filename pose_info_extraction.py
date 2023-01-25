import torch
import numpy as np
from SynergyNet.model_building import SynergyNet
from SynergyNet.FaceBoxes import FaceBoxes

class PoseInfoExtraction():
    def __init__(self):
        args = {}
        checkpoint_fp = './SynergyNet/pretrained/best.pth.tar'
        args['arch'] = 'mobilenet_v2'
        args['devices_id'] = [0]
        args['img_size'] = 120
        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        self.synergynet = SynergyNet(args)
        model_dict = self.synergynet.state_dict()
        # because the model is trained by multiple gpus, prefix 'module' should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        self.synergynet.load_state_dict(model_dict, strict=False)
        self.synergynet = self.synergynet.cuda()
        # face detector
        self.face_boxes = FaceBoxes()

    def __call__(self, x):
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x_samples = torch.zeros(x.shape[0], x.shape[1], 120, 120)
        for i in range(x.shape[0]):
            x_img = (np.moveaxis(x[i].detach().cpu().numpy(), 0, 2) * 255).astype(np.uint8)
            # rectx = self.face_boxes(x_img)
            # rect = rectx[0]
            # HCenter = (rect[1] + rect[3])/2
            # WCenter = (rect[0] + rect[2])/2
            HCenter = 64
            WCenter = 64
            Hbeginning = int(HCenter) - 60
            Hend = Hbeginning + 120
            if Hbeginning < 0:
                Hend -= Hbeginning
                Hbeginning = 0
            if Hend > 255:
                Hbeginning -= (Hend - 255)
                Hend = 255
            Wbeginning = int(WCenter) - 60
            Wend = Wbeginning + 120
            if Wbeginning < 0:
                Wend -= Wbeginning
                Wbeginning = 0
            if Wend > 255:
                Wbeginning -= (Wend - 255)
                Wend = 255
            x_samples[i] = x[i, :, Hbeginning:Hend, Wbeginning:Wend]
        x_samples = x_samples.cuda()
        x_samples = ((x_samples * 255) - 127.5) / 128
        z_x = self.synergynet.forward_test(x_samples)
        return z_x