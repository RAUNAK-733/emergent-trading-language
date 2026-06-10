"""Curriculum scheduling for offer size and auxiliary guidance."""


class CurriculumScheduler:
    """Gradually increase offer complexity and reduce early guidance."""

    def __init__(self, max_offer, strict_fraction=0.60, message_decay=0.90):
        if max_offer < 1:
            raise ValueError("max_offer must be at least 1.")
        if not 0 < strict_fraction <= 1:
            raise ValueError("strict_fraction must be in (0, 1].")
        if not 0 <= message_decay <= 1:
            raise ValueError("message_decay must be in [0, 1].")
        self.max_offer = max_offer
        self.strict_fraction = strict_fraction
        self.message_decay = message_decay

    @staticmethod
    def clamp_progress(progress):
        """Clamp arbitrary progress values to the training interval."""
        return min(max(float(progress), 0.0), 1.0)

    def offer_limit(self, progress):
        """Return the largest offer quantity currently available."""
        progress = self.clamp_progress(progress)
        return min(self.max_offer, 1 + int(progress * self.max_offer))

    def strict_weight(self, progress):
        """Return how strongly the final objective should dominate."""
        progress = self.clamp_progress(progress)
        return min(1.0, progress / self.strict_fraction)

    def message_weight(self, base_weight, progress):
        """Decay message-information regularization as training matures."""
        strict_weight = self.strict_weight(progress)
        return base_weight * (1.0 - self.message_decay * strict_weight)

    def auxiliary_weight(self, base_weight, progress):
        """Fade out the early utility-decoding auxiliary task."""
        return base_weight * (1.0 - self.strict_weight(progress))
