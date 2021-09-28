import sys
from typing import List

from digraph.digraph import Digraph
import digraph.commons as commons
from digraph.utils import load_excel
from logger import Logger
from digraph.particle import Particle

def main(argv: List[str]) -> None:
    """Entry point to the directed graph representation of combustion videos.
    
    Args:
        argv: command-line arguments
    """
    #for path in sys.path:
    #    print(path)
    Logger.set_io_level(Logger.DEBUG)
    if len(argv) < 5:
        Logger.error("Need at least four arguments: video excel, " + \
                     "x dimension, y dimension, folder for saving drawings")
        exit(1)
    particles = load_excel(argv[1])
    for p in particles:
        print(p)

    commons.PIC_DIMENSION = [int(argv[2]), int(argv[3])]
    dg = Digraph()
    dg.add_video(particles)
    dg.draw(argv[4])
    dg.draw_line_format(argv[4])
    print(dg)
    return

if __name__ == "__main__":
    main(sys.argv)