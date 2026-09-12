import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect } from 'react';

export default function LoadingScreen() {
  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(() => setVisible(false), 600);
          return 100;
        }
        const increment = prev < 60 ? 1.2 : prev < 85 ? 0.8 : 1.5;
        return Math.min(prev + increment, 100);
      });
    }, 30);
    return () => clearInterval(interval);
  }, []);

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          className="fixed inset-0 z-[99999] flex items-center justify-center overflow-hidden select-none"
          exit={{ opacity: 0, scale: 1.08, filter: 'blur(10px)' }}
          transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
        >
          {/* ===== Animated gradient background ===== */}
          <div className="absolute inset-0 bg-gradient-to-br from-sky-200 via-blue-100 to-indigo-100" />
          
          {/* Aurora bands */}
          <motion.div
            className="absolute inset-0 opacity-40"
            animate={{ backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'] }}
            transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
            style={{
              backgroundImage: 'linear-gradient(120deg, transparent 30%, rgba(56,189,248,0.3) 40%, rgba(99,102,241,0.2) 50%, rgba(34,211,238,0.3) 60%, transparent 70%)',
              backgroundSize: '200% 100%',
            }}
          />

          {/* ===== Floating clouds ===== */}
          {[
            { w: 180, h: 50, top: '12%', dur: 20, delay: 0, opacity: 0.5 },
            { w: 140, h: 40, top: '22%', dur: 25, delay: 3, opacity: 0.35 },
            { w: 200, h: 55, top: '8%', dur: 18, delay: 8, opacity: 0.4 },
            { w: 120, h: 35, top: '28%', dur: 22, delay: 12, opacity: 0.3 },
          ].map((cloud, i) => (
            <motion.div
              key={`cloud-${i}`}
              className="absolute rounded-full"
              style={{
                width: cloud.w,
                height: cloud.h,
                top: cloud.top,
                left: -cloud.w,
                opacity: cloud.opacity,
                background: 'radial-gradient(ellipse, white 0%, rgba(255,255,255,0.3) 60%, transparent 100%)',
              }}
              animate={{ x: [0, (typeof window !== 'undefined' ? window.innerWidth : 1600) + cloud.w * 2] }}
              transition={{ duration: cloud.dur, repeat: Infinity, delay: cloud.delay, ease: 'linear' }}
            />
          ))}

          {/* ===== Wind streaks ===== */}
          {[...Array(10)].map((_, i) => (
            <motion.div
              key={`wind-${i}`}
              className="absolute h-[1.5px] rounded-full"
              style={{
                width: 40 + Math.random() * 80,
                top: `${10 + i * 8}%`,
                left: '-100px',
                background: `linear-gradient(90deg, transparent, rgba(56,189,248,${0.3 + Math.random() * 0.3}), transparent)`,
              }}
              animate={{ x: [0, (typeof window !== 'undefined' ? window.innerWidth : 1600) + 200] }}
              transition={{ duration: 1.8 + Math.random() * 1.5, repeat: Infinity, delay: i * 0.35, ease: 'linear' }}
            />
          ))}

          {/* ===== Sparkle particles ===== */}
          {[...Array(20)].map((_, i) => {
            const size = 2 + Math.random() * 4;
            return (
              <motion.div
                key={`sparkle-${i}`}
                className="absolute rounded-full bg-white"
                style={{
                  width: size,
                  height: size,
                  left: `${10 + Math.random() * 80}%`,
                  top: `${10 + Math.random() * 60}%`,
                }}
                animate={{
                  opacity: [0, 1, 0],
                  scale: [0.5, 1.5, 0.5],
                }}
                transition={{
                  duration: 2 + Math.random() * 2,
                  repeat: Infinity,
                  delay: Math.random() * 4,
                  ease: 'easeInOut',
                }}
              />
            );
          })}

          {/* ===== Swirl energy particles around turbine ===== */}
          {[...Array(14)].map((_, i) => {
            const angle = (i / 14) * Math.PI * 2;
            const r = 90 + Math.random() * 50;
            return (
              <motion.div
                key={`swirl-${i}`}
                className="absolute w-2 h-2 rounded-full"
                style={{
                  left: '50%',
                  top: '38%',
                  background: `radial-gradient(circle, rgba(56,189,248,0.8), transparent)`,
                  boxShadow: '0 0 6px rgba(56,189,248,0.4)',
                }}
                animate={{
                  x: [Math.cos(angle) * 20, Math.cos(angle + 2) * r, Math.cos(angle + 4) * 30, Math.cos(angle) * 20],
                  y: [Math.sin(angle) * 20, Math.sin(angle + 2) * r, Math.sin(angle + 4) * 30, Math.sin(angle) * 20],
                  opacity: [0, 0.9, 0.5, 0],
                  scale: [0.3, 1.2, 0.6, 0.3],
                }}
                transition={{ duration: 3 + Math.random() * 2, repeat: Infinity, delay: i * 0.25, ease: 'easeInOut' }}
              />
            );
          })}

          {/* ===== Central assembly ===== */}
          <div className="relative z-10 flex flex-col items-center">
            <svg width="260" height="420" viewBox="0 0 260 420" fill="none" xmlns="http://www.w3.org/2000/svg">

              {/* Glow ring behind blades */}
              <circle cx="130" cy="115" r="85" fill="none" stroke="url(#ring-grad)" strokeWidth="1.5" opacity="0.5">
                <animate attributeName="r" values="80;90;80" dur="3s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.3;0.6;0.3" dur="3s" repeatCount="indefinite" />
              </circle>
              <circle cx="130" cy="115" r="70" fill="none" stroke="url(#ring-grad)" strokeWidth="0.8" opacity="0.3">
                <animate attributeName="r" values="65;75;65" dur="4s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.2;0.4;0.2" dur="4s" repeatCount="indefinite" />
              </circle>

              {/* ===== WINDMILL ===== */}
              {/* Tower pole with shadow */}
              <rect x="125" y="130" width="10" height="155" rx="3" fill="url(#tower-grad)" />
              <rect x="127" y="130" width="3" height="155" rx="1" fill="white" opacity="0.15" />
              
              {/* Tower base */}
              <path d="M110 285 L130 275 L150 285 Z" fill="#64748b" />
              <rect x="105" y="285" width="50" height="5" rx="2.5" fill="#475569" />
              
              {/* Ground shadow */}
              <ellipse cx="130" cy="293" rx="40" ry="4" fill="black" opacity="0.08" />

              {/* Nacelle */}
              <ellipse cx="130" cy="130" rx="12" ry="7" fill="url(#nacelle-grad)" />

              {/* ===== Rotating blades ===== */}
              <g>
                <animateTransform
                  attributeName="transform"
                  type="rotate"
                  from="0 130 125"
                  to="360 130 125"
                  dur="2.5s"
                  repeatCount="indefinite"
                />
                {/* Blade 1 — up */}
                <path d="M130 125 L126 25 Q130 8, 134 25 Z" fill="url(#blade-fill)" stroke="white" strokeWidth="0.5" opacity="0.9" />
                {/* Blade 2 — bottom-right */}
                <path d="M130 125 L200 170 Q212 178, 197 175 Z" fill="url(#blade-fill)" stroke="white" strokeWidth="0.5" opacity="0.9" />
                {/* Blade 3 — bottom-left */}
                <path d="M130 125 L60 170 Q48 178, 63 175 Z" fill="url(#blade-fill)" stroke="white" strokeWidth="0.5" opacity="0.9" />
              </g>

              {/* Static hub centre */}
              <circle cx="130" cy="125" r="9" fill="url(#hub-grad)" stroke="white" strokeWidth="1" />
              <circle cx="130" cy="125" r="5" fill="#94a3b8" />
              <circle cx="130" cy="125" r="2.5" fill="white" opacity="0.6" />

              {/* ===== WIRE ===== */}
              <path
                d="M130 290 L130 310 Q130 322, 136 328 Q142 334, 130 340 Q118 346, 130 352 L130 362"
                stroke="url(#wire-grad)"
                strokeWidth="3"
                strokeLinecap="round"
                fill="none"
              />
              {/* Wire glow */}
              <path
                d="M130 290 L130 310 Q130 322, 136 328 Q142 334, 130 340 Q118 346, 130 352 L130 362"
                stroke="rgba(56,189,248,0.2)"
                strokeWidth="8"
                strokeLinecap="round"
                fill="none"
              />

              {/* Energy pulses travelling down wire */}
              <circle r="3.5" fill="#38bdf8" filter="url(#glow)">
                <animateMotion dur="1s" repeatCount="indefinite" path="M130 290 L130 310 Q130 322, 136 328 Q142 334, 130 340 Q118 346, 130 352 L130 362" />
                <animate attributeName="opacity" values="0;1;1;0" dur="1s" repeatCount="indefinite" />
              </circle>
              <circle r="2.5" fill="#38bdf8" filter="url(#glow)" opacity="0.6">
                <animateMotion dur="1s" repeatCount="indefinite" begin="0.5s" path="M130 290 L130 310 Q130 322, 136 328 Q142 334, 130 340 Q118 346, 130 352 L130 362" />
                <animate attributeName="opacity" values="0;0.7;0.7;0" dur="1s" repeatCount="indefinite" begin="0.5s" />
              </circle>

              {/* ===== BATTERY ===== */}
              {/* Battery terminal */}
              <rect x="124" y="360" width="12" height="5" rx="2" fill="#94a3b8" />
              
              {/* Battery body */}
              <rect x="100" y="365" width="60" height="34" rx="7" fill="white" fillOpacity="0.6" stroke="#94a3b8" strokeWidth="2" />
              
              {/* Battery segments fill */}
              {[0, 1, 2, 3, 4].map((seg) => {
                const segProgress = Math.min(Math.max((progress - seg * 20) / 20, 0), 1);
                const colors = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e'];
                return (
                  <motion.rect
                    key={seg}
                    x={105 + seg * 10.5}
                    y="369"
                    width="9"
                    height="26"
                    rx="2.5"
                    fill={colors[seg]}
                    initial={{ opacity: 0, scaleY: 0 }}
                    animate={{
                      opacity: segProgress > 0 ? 0.85 : 0.1,
                      scaleY: segProgress > 0 ? 1 : 0.3,
                    }}
                    style={{ originY: '100%' }}
                    transition={{ duration: 0.3 }}
                  />
                );
              })}

              {/* Battery shimmer */}
              <rect x="100" y="365" width="60" height="34" rx="7" fill="url(#shimmer)" opacity="0.4">
                <animate attributeName="opacity" values="0.2;0.5;0.2" dur="1.5s" repeatCount="indefinite" />
              </rect>

              {/* Percentage text */}
              <text x="130" y="386" textAnchor="middle" fill="#334155" fontSize="12" fontWeight="800" fontFamily="Inter, system-ui, sans-serif">
                {Math.round(progress)}%
              </text>

              {/* ===== Gradient defs ===== */}
              <defs>
                <linearGradient id="tower-grad" x1="130" y1="130" x2="130" y2="290" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#94a3b8" />
                  <stop offset="1" stopColor="#475569" />
                </linearGradient>
                <linearGradient id="blade-fill" x1="0" y1="0" x2="0.5" y2="1">
                  <stop offset="0%" stopColor="#e2e8f0" />
                  <stop offset="50%" stopColor="#f1f5f9" />
                  <stop offset="100%" stopColor="#cbd5e1" />
                </linearGradient>
                <radialGradient id="hub-grad" cx="0.4" cy="0.35" r="0.6">
                  <stop offset="0%" stopColor="#cbd5e1" />
                  <stop offset="100%" stopColor="#64748b" />
                </radialGradient>
                <radialGradient id="nacelle-grad" cx="0.4" cy="0.3" r="0.7">
                  <stop offset="0%" stopColor="#e2e8f0" />
                  <stop offset="100%" stopColor="#94a3b8" />
                </radialGradient>
                <linearGradient id="wire-grad" x1="130" y1="290" x2="130" y2="362" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#64748b" />
                  <stop offset="1" stopColor="#94a3b8" />
                </linearGradient>
                <linearGradient id="ring-grad" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.6" />
                  <stop offset="50%" stopColor="#818cf8" stopOpacity="0.3" />
                  <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.6" />
                </linearGradient>
                <linearGradient id="shimmer" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="white" stopOpacity="0" />
                  <stop offset="50%" stopColor="white" stopOpacity="0.5" />
                  <stop offset="100%" stopColor="white" stopOpacity="0" />
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
            </svg>

            {/* ===== Text & bolt below ===== */}
            <div className="flex flex-col items-center gap-3 -mt-4">
              {/* Lightning bolt */}
              <motion.div
                animate={{ opacity: [0.5, 1, 0.5], scale: [0.9, 1.1, 0.9], y: [0, -3, 0] }}
                transition={{ duration: 1, repeat: Infinity, ease: 'easeInOut' }}
              >
                <svg width="24" height="28" viewBox="0 0 20 24" fill="none">
                  <path d="M11 1L3 14H10L9 23L17 10H10L11 1Z" fill="url(#bolt-grad)" stroke="#eab308" strokeWidth="0.8" strokeLinejoin="round" />
                  <defs>
                    <linearGradient id="bolt-grad" x1="10" y1="0" x2="10" y2="24">
                      <stop stopColor="#fde68a" />
                      <stop offset="1" stopColor="#f59e0b" />
                    </linearGradient>
                  </defs>
                </svg>
              </motion.div>

              {/* Brand text */}
              <motion.div
                className="flex flex-col items-center gap-1"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.6 }}
              >
                <h2 className="text-lg font-bold tracking-wider bg-gradient-to-r from-blue-600 via-sky-500 to-cyan-500 bg-clip-text text-transparent">
                  ZERO BIAS
                </h2>
                <motion.p
                  className="text-sm font-medium text-slate-400 tracking-wide"
                  animate={{ opacity: [0.4, 1, 0.4] }}
                  transition={{ duration: 2.5, repeat: Infinity }}
                >
                  {progress < 100 ? 'Reading the weather models…' : 'Forecast ready'}
                </motion.p>
              </motion.div>
            </div>
          </div>

          {/* ===== Grass / ground landscape ===== */}
          <div className="absolute bottom-0 left-0 right-0 h-24 pointer-events-none z-0">
            <svg viewBox="0 0 1440 100" preserveAspectRatio="none" className="w-full h-full">
              <path d="M0 40 Q 180 10, 360 35 T 720 30 T 1080 38 T 1440 25 L1440 100 L0 100 Z" fill="#86efac" opacity="0.3" />
              <path d="M0 55 Q 200 30, 400 50 T 800 45 T 1200 55 T 1440 40 L1440 100 L0 100 Z" fill="#4ade80" opacity="0.25" />
              <path d="M0 70 Q 240 50, 480 65 T 960 60 T 1440 55 L1440 100 L0 100 Z" fill="#22c55e" opacity="0.15" />
            </svg>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
