import numpy as np
import math
import torch
import scipy
import imquality.brisque as brisque
import PIL.Image
from .vif_utils import *
from skimage.metrics import structural_similarity as ssim

class PerformanceMetrics():
    """
    Class for computing performance scores
    Usage:
    0. instantiate metrics object with correction for image scale img_scaling=scaling_kspace/scaling_smaps
    1. clear the sample storage: `metrics.clear()`
    2. add samples to the storage: `metrics.add(img, ref)`
    3. evaluate metrics on all samples in storage: `metrics.avg_ssid()`
    """
    def __init__(self, img_scaling=1., ssim=True, psnr=True, ser=True, hfen=True, brisque=True, vif=True, mse=True, crossection_vif=True, crosssection_vif_roi=[[50, 150], [70, 150]]):
        self.samples = []
        self.img_scaling = img_scaling
        self.crosssection_vif_roi = crosssection_vif_roi
        self.clear_history()
        self.active_metrics = {
            "ssim": ssim,
            "psnr": psnr,
            "ser": ser,
            "hfen": hfen,
            "brisque": brisque,
            "vif": vif,
            "mse": mse,
            "crossection_vif": crossection_vif
        }

    ## Sample storage
    def add(self, img, ref, index=None):

        assert len(img.shape) == 2 # only consider real (not complex) images
        assert len(ref.shape) == 2

        img_copy = torch.clone(img.detach()) / self.img_scaling # reverse the scaling of the img
        ref_copy = torch.clone(ref.detach())

        self.samples.append({
            "img": img_copy.cpu(),
            "ref": ref_copy.cpu(),
            "index": index
        })
    
    def clear(self):
        self.samples = []

    ## History
    def save_all_to_history(self, iteration):
        self.history["iterations"].append(iteration)
        if self.active_metrics["ssim"]:
            self.history["ssim"].append(self.compute_score_for_all_samples(self.ssim))
        if self.active_metrics["psnr"]:
            self.history["psnr"].append(self.compute_score_for_all_samples(self.psnr))
        if self.active_metrics["ser"]:
            self.history["ser"].append(self.compute_score_for_all_samples(self.ser))
        if self.active_metrics["hfen"]:
            self.history["hfen"].append(self.compute_score_for_all_samples(self.hfen))
        if self.active_metrics["brisque"]:
            self.history["brisque"].append(self.compute_score_for_all_samples(self.brisque))
        if self.active_metrics["vif"]:
            self.history["vif"].append(self.compute_score_for_all_samples(self.vif))
        if self.active_metrics["mse"]:
            self.history["mse"].append(self.compute_score_for_all_samples(self.mse))
        if self.active_metrics["crossection_vif"]:
            self.history["crossection_vif"].append(self.compute_crossection_vif_accross_all_samples())

    def clear_history(self):
        self.history = {
            "iterations": [],
            "ssim": [],
            "psnr": [],
            "hfen": [],
            "ser": [],
            "vif": [],
            "mse": [],
            "brisque": [],
            "crossection_vif": []
        }

    def save_history_to_file(self, path):
        torch.save(self.history, path)

    def load_history_from_file(self, path):
        self.history = torch.load(path)

    def get_history_averages(self):
        averages = dict()
        for key, value in self.history.items():
            averages[key] = np.zeros(len(value))
            for i in range(len(value)):
                averages[key][i] = np.mean(np.array(self.history[key][i]))
        return averages

    def get_history_std(self):
        stds = dict()
        for key, value in self.history.items():
            stds[key] = np.zeros(len(value))
            for i in range(len(value)):
                stds[key][i] = np.std(np.array(self.history[key][i]))
        return stds

    ## Helpers for easy access 
    def compute_score_for_all_samples(self, score_fct):
        values = torch.zeros(len(self.samples))
        for i, s in enumerate(self.samples):
            values[i] = score_fct(s["img"], s["ref"])
        return values

    def compute_ssims(self):
        return self.compute_score_for_all_samples(self.ssim)

    def compute_psnrs(self):
        return self.compute_score_for_all_samples(self.psnr)

    def compute_sers(self):
        return self.compute_score_for_all_samples(self.ser)

    def compute_hfens(self):
        return self.compute_score_for_all_samples(self.hfen)

    def compute_brisques(self):
        return self.compute_score_for_all_samples(self.brisque)

    def compute_vifs(self):
        return self.compute_score_for_all_samples(self.vif)

    def avg_ssim(self):
        return torch.mean(self.compute_score_for_all_samples(self.ssim))
    
    def avg_psnr(self):
        return torch.mean(self.compute_score_for_all_samples(self.psnr))

    def avg_vif(self):
        return torch.mean(self.compute_score_for_all_samples(self.vif))

    def avg_mse(self):
        return torch.mean(self.compute_score_for_all_samples(self.mse))

    def compute_crossection_vif_accross_all_samples(self):
        ref_imgs = [sample["ref"] for sample in self.samples]
        imgs = [sample["img"] for sample in self.samples]
        return self.crosssection_vif(imgs, ref_imgs)

    ## Metrics
    # structural similarity index
    @staticmethod
    def ssim(img, ref):
        # img and ref shape: (Ny, Nx)

        if isinstance(img, np.ndarray):
            img = torch.tensor(img)
        if isinstance(ref, np.ndarray):
            ref = torch.tensor(ref)

        max_value = float(ref.max())
        min_value = float(ref.min())

        img_clamped = img.clone()
        img_clamped[img_clamped > max_value] = max_value

        return ssim(ref.cpu().numpy(), img_clamped.cpu().numpy(), data_range=max_value - min_value)
        # return structural_similarity(ref.cpu().numpy(), img.cpu().numpy(), data_range=max_value, gaussian_weights=True, sigma=1.5)


    # Signal-to-Error Ratio in dB
    # \mathrm{SER}=20 \operatorname{log}_{10} \frac{\left\|\mathbf{x}_{\text {orig }}\right\|}{\left\|\mathbf{x}_{\text {orig }}-\mathbf{x}_{\text {recon }}\right\|}
    @classmethod
    def ser(self, img, ref):

        if isinstance(img, np.ndarray):
            img = torch.tensor(img)
        if isinstance(ref, np.ndarray):
            ref = torch.tensor(ref)

        # img and ref shape: (Ny, Nx)
        return 20. * torch.log(torch.linalg.norm(ref) / (torch.linalg.norm(img - ref))) / math.log(10.)
    
    # Peak-Signal-to-Noise Ratio in dB
    @classmethod
    def psnr(self, img, ref, max_value=None):

        if isinstance(img, np.ndarray):
            img = torch.tensor(img)
        if isinstance(ref, np.ndarray):
            ref = torch.tensor(ref)

        # img and ref shape: (Ny, Nx)
        if max_value is None:
            max_value = torch.abs(ref).max()

        mse = self.mse(img, ref)
        return 10. * torch.log(max_value**2 / mse) / math.log(10.)

    @classmethod
    def hfen(self, img, ref):

        if isinstance(img, np.ndarray):
            img = torch.tensor(img)
        if isinstance(ref, np.ndarray):
            ref = torch.tensor(ref)
            
        img_log = scipy.ndimage.gaussian_laplace(img.cpu().numpy(), sigma=1.5, mode='reflect')
        ref_log = scipy.ndimage.gaussian_laplace(ref.cpu().numpy(), sigma=1.5, mode='reflect')
        return 20. * math.log(np.linalg.norm(ref_log - img_log) / np.linalg.norm(ref_log)) / math.log(10.)

    # Mean Squared Error
    @classmethod
    def mse(self, img, ref):

        if isinstance(img, np.ndarray):
            img = torch.tensor(img)
        if isinstance(ref, np.ndarray):
            ref = torch.tensor(ref)

        return torch.mean(torch.square(img.flatten() - ref.flatten()))

    @classmethod
    def brisque(self, img, ref):

        if torch.is_tensor(img):
            img = img.cpu().numpy()

        img = np.abs(img)
        im = PIL.Image.fromarray(img / np.max(img))
        im = im.convert('RGB')
        
        try:
            return brisque.score(im)
        except:
            return float("NAN")

    def vif(self, img, ref):

        if torch.is_tensor(img):
            img = img.cpu().numpy()
        if torch.is_tensor(ref):
            ref = ref.cpu().numpy()

        img = np.abs(img)
        ref = np.abs(ref)

        normalization_factor = 255 / np.max(ref)
        try:
            return vif(ref * normalization_factor, img * normalization_factor)
        except Exception as e:
            # print(e)
            return float("NaN")

    def crosssection_vif(self, imgs, ref_imgs):

        if not torch.is_tensor(imgs[0]):
            imgs = torch.stack(imgs, axis=0)
        else:
            imgs = np.stack(imgs, axis=0)

        if torch.is_tensor(ref_imgs[0]):
            ref_imgs = torch.stack(ref_imgs, axis=0)
        else:
            ref_imgs = np.stack(ref_imgs, axis=0)

        roi = self.crosssection_vif_roi

        vif_scores = []
        for x in range(roi[0][0], roi[0][1]):
            vif_scores.append(self.vif(imgs[:, x, roi[1][0]:roi[1][1]], ref_imgs[:, x, roi[1][0]:roi[1][1]]))
        for y in range(roi[1][0], roi[1][1]):
            vif_scores.append(self.vif(imgs[:, roi[0][0]:roi[0][1], y], ref_imgs[:, roi[0][0]:roi[0][1], y]))

        return np.mean(np.array(vif_scores))

    

