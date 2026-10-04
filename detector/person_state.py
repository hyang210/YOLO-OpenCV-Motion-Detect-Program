"""추적 ID 별 상태: 베이스라인, 연속 프레임 확정, 이벤트 쿨다운"""
from dataclasses import dataclass, field

from .config import FrameParams


@dataclass
class PersonState:
    params: FrameParams
    sum_knee: float = 0.0
    sum_hip: float = 0.0
    n_base: int = 0
    streak_label: str = 'Normal'
    streak: int = 0
    status: str = 'Normal'
    last_seen: int = 0
    last_event: dict = field(default_factory=dict)

    @property
    def baseline(self):
        if self.n_base < self.params.baseline:
            return None
        return self.sum_knee / self.n_base, self.sum_hip / self.n_base

    def add_baseline(self, f):
        if self.n_base < self.params.baseline and f.knee is not None and not f.foot_above_knee:
            self.sum_knee += f.knee
            self.sum_hip += f.hip
            self.n_base += 1

    def update(self, label):
        """연속 프레임 확인. 확정된 이벤트면 이벤트명을, 아니면 None 을 반환."""
        if label == 'Normal':
            self.streak_label, self.streak, self.status = 'Normal', 0, 'Normal'
            return None
        self.streak = self.streak + 1 if label == self.streak_label else 1
        self.streak_label = label
        if self.streak >= self.params.confirm:
            self.status = label
            return label
        return None

    def should_log(self, event, frame_num):
        """쿨다운이 지났으면 기록 시점을 갱신하고 True."""
        last = self.last_event.get(event)
        if last is not None and frame_num - last < self.params.cooldown:
            return False
        self.last_event[event] = frame_num
        return True
