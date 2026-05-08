import { openDB, type IDBPDatabase } from 'idb';
import type { Trip } from '../types';

interface Schema {
  trips: { key: string; value: Trip };
}

let dbp: Promise<IDBPDatabase<Schema>> | null = null;

function db() {
  if (!dbp) {
    dbp = openDB<Schema>('drowsy-app', 1, {
      upgrade(d) {
        d.createObjectStore('trips', { keyPath: 'id' });
      },
    });
  }
  return dbp;
}

export async function saveTrip(trip: Trip): Promise<void> {
  const d = await db();
  await d.put('trips', trip);
}

export async function listTrips(): Promise<Trip[]> {
  const d = await db();
  const all = await d.getAll('trips');
  return all.sort((a, b) => b.startedAt - a.startedAt);
}

export async function clearTrips(): Promise<void> {
  const d = await db();
  await d.clear('trips');
}
