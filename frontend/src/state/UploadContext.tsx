import { createContext, ReactNode, useContext, useMemo, useState } from "react";

interface UploadState {
  pvId?: string;
  cycleId?: string;
}

interface UploadContextValue {
  uploads: UploadState;
  setUploads: (next: UploadState) => void;
  update: (partial: Partial<UploadState>) => void;
  clear: () => void;
}

const UploadContext = createContext<UploadContextValue | undefined>(undefined);

export function UploadProvider({ children }: { children: ReactNode }) {
  const [uploads, setUploads] = useState<UploadState>(() => {
    const cached = typeof window !== "undefined" ? window.localStorage.getItem("besslab-uploads") : null;
    return cached ? (JSON.parse(cached) as UploadState) : {};
  });

  const persist = (next: UploadState) => {
    setUploads(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem("besslab-uploads", JSON.stringify(next));
    }
  };

  const value = useMemo<UploadContextValue>(
    () => ({
      uploads,
      setUploads: persist,
      update: (partial) => persist({ ...uploads, ...partial }),
      clear: () => persist({}),
    }),
    [uploads]
  );

  return <UploadContext.Provider value={value}>{children}</UploadContext.Provider>;
}

export function useUploads() {
  const ctx = useContext(UploadContext);
  if (!ctx) {
    throw new Error("UploadContext is missing");
  }
  return ctx;
}
