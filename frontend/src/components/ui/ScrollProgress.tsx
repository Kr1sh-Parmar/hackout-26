import { motion, useScroll } from 'framer-motion';
import { RefObject } from 'react';

interface Props {
  containerRef: RefObject<HTMLElement>;
}

export default function ScrollProgress({ containerRef }: Props) {
  // The ref is owned by the parent; without layoutEffect:false framer reads it before it is attached.
  const { scrollYProgress } = useScroll({ container: containerRef as RefObject<HTMLElement>, layoutEffect: false });

  return (
    <motion.div
      className="absolute top-0 left-0 right-0 h-[3px] bg-gradient-to-r from-blue-500 via-cyan-400 to-blue-500 origin-left z-50 rounded-full"
      style={{ scaleX: scrollYProgress }}
    />
  );
}
