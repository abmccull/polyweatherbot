import type { Season, WorldClock } from "../state/gameState";

const SEASONS: Season[] = ["Spring", "Summer", "Autumn", "Winter"];

export function getAbsoluteDay(clock: Pick<WorldClock, "tick" | "ticksPerDay">): number {
  return Math.floor(clock.tick / clock.ticksPerDay) + 1;
}

export function advanceClock(clock: WorldClock, ticks: number): { clock: WorldClock; dayChanged: boolean; monthChanged: boolean } {
  let nextClock = { ...clock };
  let dayChanged = false;
  let monthChanged = false;

  for (let index = 0; index < ticks; index += 1) {
    nextClock = {
      ...nextClock,
      tick: nextClock.tick + 1,
      tickOfDay: nextClock.tickOfDay + 1
    };

    if (nextClock.tickOfDay >= nextClock.ticksPerDay) {
      dayChanged = true;
      nextClock = {
        ...nextClock,
        tickOfDay: 0,
        day: nextClock.day + 1
      };

      if (nextClock.day > 30) {
        monthChanged = true;
        const nextMonth = nextClock.month + 1;
        const normalizedMonth = nextMonth > 12 ? 1 : nextMonth;
        const nextYear = nextMonth > 12 ? nextClock.year + 1 : nextClock.year;
        nextClock = {
          ...nextClock,
          day: 1,
          month: normalizedMonth,
          year: nextYear,
          season: SEASONS[Math.floor((normalizedMonth - 1) / 3)] ?? "Spring"
        };
      }
    }
  }

  return {
    clock: nextClock,
    dayChanged,
    monthChanged
  };
}
