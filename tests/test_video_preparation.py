"""Display-frame indexing and cadence must not trust MP4 sample-count metadata."""
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'comfyui_vipe_remote'))
from worker import decoded_video_info, prepare_plate


def probe_fixture(ticks, avg='520/9', nominal='60/1', time_base='1/12000', stored='338'):
    return dict(streams=[dict(nb_frames=stored, avg_frame_rate=avg, r_frame_rate=nominal, time_base=time_base)], frames=[dict(best_effort_timestamp=t) for t in ticks])


class VideoPreparationTests(unittest.TestCase):
    def test_mp4_display_count_and_cadence_override_stored_samples(self):
        info=decoded_video_info(probe_fixture([i*200 for i in range(323)]))
        self.assertEqual(info['nb_frames'],'323')
        self.assertEqual(info['reported_nb_frames'],'338')
        self.assertEqual(Fraction(info['avg_frame_rate']),60)
        self.assertEqual(info['reported_avg_frame_rate'],'520/9')
        self.assertEqual(info['frame_rate_method'],'decoded_constant_cadence')

    def test_fractional_rate_remains_exact(self):
        info=decoded_video_info(probe_fixture([i*1001 for i in range(20)],avg='30000/1001',nominal='30000/1001',time_base='1/30000'))
        self.assertEqual(Fraction(info['avg_frame_rate']),Fraction(30000,1001))

    def test_coarse_timestamp_quantization(self):
        info=decoded_video_info(probe_fixture([round(i*1000/60) for i in range(20)],time_base='1/1000'))
        self.assertEqual(Fraction(info['avg_frame_rate']),60)

    def test_variable_rate_does_not_blindly_use_nominal_rate(self):
        info=decoded_video_info(probe_fixture([0,200,800,1000,1400]))
        self.assertEqual(Fraction(info['avg_frame_rate']),Fraction(520,9))
        self.assertEqual(info['frame_rate_method'],'average_rate_normalization')

    def test_coarse_time_base_does_not_accept_accumulated_timing_error(self):
        info=decoded_video_info(probe_fixture([i*2 for i in range(20)],avg='30/1',nominal='60/1',time_base='1/60'))
        self.assertEqual(Fraction(info['avg_frame_rate']),30)
        self.assertEqual(info['frame_rate_method'],'average_rate_normalization')

    def test_missing_timestamps_still_uses_decoded_count(self):
        data=probe_fixture([0,200,400]);data['frames'][1]={}
        info=decoded_video_info(data)
        self.assertEqual(info['nb_frames'],'3')
        self.assertEqual(Fraction(info['avg_frame_rate']),Fraction(520,9))

    def test_invalid_average_can_use_valid_nominal(self):
        info=decoded_video_info(probe_fixture([0,200],avg='0/0'))
        self.assertEqual(Fraction(info['avg_frame_rate']),60)

    def test_no_decodable_frames_is_actionable(self):
        with self.assertRaisesRegex(ValueError,'no decodable display frames'):
            decoded_video_info(probe_fixture([]))

    def test_output_rate_guard(self):
        with patch('worker.subprocess.run'),patch('worker.probe_video',return_value={'nb_frames':'3','avg_frame_rate':'30/1'}):
            with self.assertRaisesRegex(ValueError,'changed the frame rate'):
                prepare_plate('source','output',dict(ffmpeg='ffmpeg',ffprobe='ffprobe'),0,3,Fraction(60),64,32)

    def test_output_count_guard_is_retained(self):
        with patch('worker.subprocess.run'),patch('worker.probe_video',return_value={'nb_frames':'2'}):
            with self.assertRaisesRegex(ValueError,'produced 2 decoded frames; expected 3'):
                prepare_plate('source','output',dict(ffmpeg='ffmpeg',ffprobe='ffprobe'),0,3,Fraction(60),64,32)


if __name__=='__main__':unittest.main()
