import { useEffect, useRef, useState } from 'react';
import { motion, useInView } from 'framer-motion';

interface Props {
  value: string;
  className?: string;
}

export default function AnimatedCounter({ value, className = '' }: Props) {
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true });
  const [display, setDisplay] = useState('0');

  // Extract numeric part and any prefix/suffix
  const numericMatch = value.match(/^([^0-9]*)([0-9,.]+)(.*)$/);
  const prefix = numericMatch?.[1] || '';
  const numStr = numericMatch?.[2] || value;
  const suffix = numericMatch?.[3] || '';
  const target = parseFloat(numStr.replace(/,/g, ''));
  const hasCommas = numStr.includes(',');
  const decimals = numStr.includes('.') ? numStr.split('.')[1].length : 0;

  useEffect(() => {
    if (!isInView) return;
    
    const duration = 1800; // ms
    const startTime = Date.now();

    const tick = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = eased * target;

      let formatted = decimals > 0
        ? current.toFixed(decimals)
        : Math.round(current).toString();
      
      if (hasCommas) {
        const parts = formatted.split('.');
        parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        formatted = parts.join('.');
      }

      setDisplay(formatted);

      if (progress < 1) {
        requestAnimationFrame(tick);
      }
    };

    requestAnimationFrame(tick);
  }, [isInView, target, decimals, hasCommas]);

  return (
    <motion.span
      ref={ref}
      className={className}
      initial={{ opacity: 0, y: 10 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5 }}
    >
      {prefix}{display}{suffix}
    </motion.span>
  );
}
