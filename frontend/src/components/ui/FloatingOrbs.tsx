import { motion } from 'framer-motion';

const orbs = [
  { size: 300, x: '10%', y: '20%', color: 'from-blue-400/20 to-cyan-300/10', duration: 18 },
  { size: 200, x: '70%', y: '60%', color: 'from-green-400/15 to-emerald-300/10', duration: 22 },
  { size: 250, x: '80%', y: '10%', color: 'from-orange-300/15 to-yellow-200/10', duration: 20 },
  { size: 180, x: '30%', y: '75%', color: 'from-purple-400/10 to-pink-300/5', duration: 25 },
];

export default function FloatingOrbs() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
      {orbs.map((orb, i) => (
        <motion.div
          key={i}
          className={`absolute rounded-full bg-gradient-to-br ${orb.color} blur-3xl`}
          style={{ width: orb.size, height: orb.size, left: orb.x, top: orb.y }}
          animate={{
            x: [0, 40, -30, 20, 0],
            y: [0, -30, 20, -40, 0],
            scale: [1, 1.15, 0.9, 1.1, 1],
          }}
          transition={{
            duration: orb.duration,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  );
}
