from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from PIL import Image
import random

from flatland.utils.rendertools import RenderTool
from IPython.display import clear_output, display
import PIL

class GifCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    gif = []
    record = False
    env_renderer = None

    def __init__(self, verbose=0):
        super(GifCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        if random.random() <= 0.25:
            print('Recording')
            self.record = True

        if self.record == True:    
            self.gif = []
            self.env_renderer = RenderTool(self.training_env, gl="PILSVG")
            print('true')

        pass

    def _on_step(self) -> bool:

        if self.record == True:
            self.env_renderer.render_env()

            image = self.env_renderer.get_image()
            pil_image = Image.fromarray(image)
            self.gif.append(pil_image)

        return True

    def _on_rollout_end(self) -> None:
        if self.record == True:
            self.gif[0].save('runs/step_ ' + str(self.n_calls) + '.gif', save_all=True,optimize=False, append_images=self.gif[1:], loop=0)
            self.gif = []
            print('Saving Gif')
            self.record = False
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass




    # def render_env(env,wait=True):
    
#     env_renderer = RenderTool(env, gl="PILSVG")
#     env_renderer.render_env()

#     image = env_renderer.get_image()
#     pil_image = PIL.Image.fromarray(image)
#     clear_output(wait=True)
#     # display(pil_image)
#     pil_image.show()
