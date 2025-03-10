import torch
# calibration error
class CalibrationError(torch.nn.Module):
    """
    Credits to: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    Calculates the Expected Calibration Error & Maximum Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(CalibrationError, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        labels = torch.from_numpy(labels)
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece  = torch.zeros(1, device=logits.device)
        sece = torch.zeros(1, device=logits.device)
        mce = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece  += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                sece += (avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                mce.append(torch.abs(avg_confidence_in_bin - accuracy_in_bin))
        else:
            try:
                mce = torch.stack(mce).max()
            except:
                mce = torch.tensor([1])
        return ece, sece, mce
    
    def plot_calibration(self):
        pass

if __name__ == '__main__':
    cali = CalibrationError(n_bins=10)